"""Typed PP-OCRv6 vendor observation boundary for routing and export projections."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.api_image_codec import encode_image_bytes_for_paddle
from app.core.bbox_extraction import bbox_from_variant
from app.core.paddle_v16_client import PaddleV16LayoutClient
from app.models.charocr_routing import (
    TEXT_AXIS_HORIZONTAL,
    TEXT_AXIS_VERTICAL,
    xyxy,
)


PPOCR_V6_MODEL = "PP-OCRv6"
VALID_TEXT_AXES = frozenset({TEXT_AXIS_HORIZONTAL, TEXT_AXIS_VERTICAL})


@dataclass(frozen=True)
class PpOcrV6RoutingRequestProfile:
    """Explicit vendor request contract for the routing prepass."""

    model: str
    options: tuple[tuple[str, object], ...]

    def optional_payload(self) -> dict[str, object]:
        return dict(self.options)


def build_ppocr_v6_routing_request_profile() -> PpOcrV6RoutingRequestProfile:
    """Return the PP-OCRv6 request profile for CharOCR routing.

    Detection geometry stays owned by the deployed PP-OCRv6 model.  The only
    detector override is the box acceptance threshold validated across the
    routing stress corpus; side-length, pixel, and unclip settings remain at
    the model defaults.
    """
    return PpOcrV6RoutingRequestProfile(
        model=PPOCR_V6_MODEL,
        options=(
            ("useDocOrientationClassify", False),
            ("useDocUnwarping", False),
            ("useTextlineOrientation", True),
            ("returnWordBox", True),
            ("textDetBoxThresh", 0.4),
            ("textRecScoreThresh", 0.0),
        ),
    )


@dataclass(frozen=True)
class PpOcrV6WordBox:
    """One PP-OCRv6 word/digit/punctuation proposal inside a physical line."""

    line_index: int
    token_index: int
    text: str
    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("PP-OCRv6 word box requires non-empty geometry")


@dataclass(frozen=True)
class PpOcrV6LineHint:
    """A PP-OCRv6 physical row plus its optional word-box proposals."""

    index: int
    text: str
    bbox: tuple[int, int, int, int]
    words: tuple[PpOcrV6WordBox, ...]
    polygon: tuple[tuple[int, int], ...] = ()
    text_axis: str = TEXT_AXIS_HORIZONTAL
    orientation_angle: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(
            self,
            "polygon",
            tuple((int(point[0]), int(point[1])) for point in self.polygon),
        )
        axis = str(self.text_axis or TEXT_AXIS_HORIZONTAL).strip().lower()
        if axis not in VALID_TEXT_AXES:
            raise ValueError(f"unsupported PP-OCRv6 text axis: {axis!r}")
        object.__setattr__(self, "text_axis", axis)
        angle = int(self.orientation_angle)
        if angle not in {-1, 0, 180}:
            raise ValueError(f"unsupported PP-OCRv6 textline orientation angle: {angle}")
        object.__setattr__(self, "orientation_angle", angle)
        if any(not isinstance(word, PpOcrV6WordBox) for word in self.words):
            raise TypeError("PP-OCRv6 line word boxes require typed observations")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("PP-OCRv6 line hint requires non-empty geometry")


@dataclass(frozen=True)
class PpOcrV6PrepassArtifact:
    """External PP-OCRv6 observation for one page and one request run.

    It is read-only vendor observation, not a source of truth for layout or
    proof. The routing compiler may derive CharOCR routes from it, and the
    export-owned table text-layer projection may derive hidden cell geometry
    from the same observation without mutating the observation itself.
    """

    page_uid: str
    run_id: str
    lines: tuple[PpOcrV6LineHint, ...]

    def __post_init__(self) -> None:
        if not self.page_uid:
            raise ValueError("PP-OCRv6 prepass artifact requires page_uid")
        object.__setattr__(self, "lines", tuple(self.lines))
        if any(not isinstance(line, PpOcrV6LineHint) for line in self.lines):
            raise TypeError("PP-OCRv6 prepass artifact lines require typed observations")


class PpOcrV6PrepassClient:
    """Use the common Paddle jobs transport with the PP-OCRv6 model contract."""

    def __init__(self, transport: PaddleV16LayoutClient) -> None:
        self._transport = transport
        self._request_profile = build_ppocr_v6_routing_request_profile()

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
            model=self._request_profile.model,
            optional_payload=self._request_profile.optional_payload(),
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
    return normalize_ppocr_v6_prepass_result(
        result_items[0],
        page_uid=page_uid,
        run_id=run_id,
        width=width,
        height=height,
    )


def normalize_ppocr_v6_prepass_result(
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
    polygons = pruned.get("rec_polys", [])
    orientation_angles = pruned.get("textline_orientation_angles", [])
    words_by_line = pruned.get("text_word", [])
    word_boxes_by_line = pruned.get("text_word_boxes", [])
    if not isinstance(texts, list) or not isinstance(boxes, list):
        raise ValueError("PP-OCRv6 result missing rec_texts/rec_boxes lists")
    if len(texts) != len(boxes):
        raise ValueError(
            "PP-OCRv6 result has mismatched rec_texts/rec_boxes lengths: "
            f"{len(texts)} != {len(boxes)}"
        )
    if polygons is not None and not isinstance(polygons, list):
        raise ValueError("PP-OCRv6 result rec_polys must be a list")
    if orientation_angles is not None and not isinstance(orientation_angles, list):
        raise ValueError("PP-OCRv6 result textline_orientation_angles must be a list")
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
        if raw_words and _normalize_text_stream(raw_text) != _normalize_text_stream("".join(str(word or "") for word in raw_words)):
            raise ValueError(
                f"PP-OCRv6 line {line_index} word tokens do not reproduce its line text"
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
        polygon = _parse_polygon(
            polygons[line_index] if line_index < len(polygons) else None,
            width=width,
            height=height,
        )
        orientation_angle = _parse_orientation_angle(
            orientation_angles[line_index]
            if line_index < len(orientation_angles)
            else -1
        )
        lines.append(PpOcrV6LineHint(
            index=line_index,
            text=str(raw_text or ""),
            bbox=line_bbox,
            words=tuple(words),
            polygon=polygon,
            text_axis=_text_axis_from_polygon_or_bbox(polygon, line_bbox),
            orientation_angle=orientation_angle,
        ))
    return PpOcrV6PrepassArtifact(page_uid=page_uid, run_id=run_id, lines=tuple(lines))


def _parse_bbox(value: object, *, width: int | None, height: int | None) -> tuple[int, int, int, int] | None:
    bbox = bbox_from_variant(value, max_w=width, max_h=height)
    if bbox is None or bbox.area <= 0:
        return None
    return bbox.to_xyxy()


def _parse_polygon(
    value: object,
    *,
    width: int | None,
    height: int | None,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return ()
    points: list[tuple[int, int]] = []
    for raw_point in value[:4]:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            return ()
        x = int(round(float(raw_point[0])))
        y = int(round(float(raw_point[1])))
        if width is not None:
            x = min(max(0, x), int(width))
        if height is not None:
            y = min(max(0, y), int(height))
        points.append((x, y))
    return tuple(points)


def _text_axis_from_polygon_or_bbox(
    polygon: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
) -> str:
    if len(polygon) == 4:
        horizontal_edge = max(
            _squared_distance(polygon[0], polygon[1]),
            _squared_distance(polygon[2], polygon[3]),
        )
        vertical_edge = max(
            _squared_distance(polygon[1], polygon[2]),
            _squared_distance(polygon[3], polygon[0]),
        )
        if horizontal_edge != vertical_edge:
            return (
                TEXT_AXIS_HORIZONTAL
                if horizontal_edge > vertical_edge
                else TEXT_AXIS_VERTICAL
            )
    return (
        TEXT_AXIS_HORIZONTAL
        if bbox[2] - bbox[0] >= bbox[3] - bbox[1]
        else TEXT_AXIS_VERTICAL
    )


def _squared_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _parse_orientation_angle(value: object) -> int:
    if value is None:
        return -1
    try:
        angle = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid PP-OCRv6 textline orientation angle: {value!r}"
        ) from exc
    if angle not in {-1, 0, 180}:
        raise ValueError(f"unsupported PP-OCRv6 textline orientation angle: {angle}")
    return angle


def _normalize_text_stream(value: object) -> str:
    return "".join(str(value or "").split())


__all__ = [
    "PPOCR_V6_MODEL",
    "TEXT_AXIS_HORIZONTAL",
    "TEXT_AXIS_VERTICAL",
    "PpOcrV6LineHint",
    "PpOcrV6PrepassArtifact",
    "PpOcrV6PrepassClient",
    "PpOcrV6RoutingRequestProfile",
    "PpOcrV6WordBox",
    "build_ppocr_v6_routing_request_profile",
    "normalize_ppocr_v6_prepass_result",
    "parse_ppocr_v6_prepass_jsonl",
]
