"""Typed observations from Paddle's real inline-formula detector payload."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

from app.adapters.paddle.layout_importer import map_paddle_label_to_block_type
from app.core.paddle_labels import normalize_paddle_label
from app.core.text_classification import is_formula_marker_token
from app.models.enums import BlockType
from app.models.geometry import BBox


_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)
_NON_TEXT_PARENT_TYPES = {
    BlockType.EQUATION,
    BlockType.FIGURE,
    BlockType.TABLE,
    BlockType.UNKNOWN,
}


class PaddleInlineFormulaObservationError(ValueError):
    """The vendor payload contains malformed inline-formula geometry."""


@dataclass(frozen=True, slots=True)
class PaddleInlineFormulaDetectorObservation:
    item_index: int
    detector_index: int
    bbox: BBox
    score: float | None
    raw_json_path: str
    parent_raw_index: int | None = None
    parent_json_path: str = ""
    parent_text: str = ""
    exact_parent_text: str = ""


@dataclass(frozen=True, slots=True)
class _ParentRecord:
    raw_index: int
    json_path: str
    bbox: BBox
    text: str


def inline_formula_detector_observations(
    response: Mapping[str, Any],
    *,
    page_width: int,
    page_height: int,
) -> tuple[PaddleInlineFormulaDetectorObservation, ...]:
    """Read detector boxes without promoting them into layout or OCR truth."""
    result = response.get("result")
    if not isinstance(result, Mapping):
        return ()
    items = result.get("layoutParsingResults")
    if not isinstance(items, list):
        return ()

    observations: list[PaddleInlineFormulaDetectorObservation] = []
    parents: list[_ParentRecord] = []
    raw_index = 0
    for item_index, item_value in enumerate(items):
        if not isinstance(item_value, Mapping):
            continue
        pruned = item_value.get("prunedResult")
        if not isinstance(pruned, Mapping):
            continue
        scale_x, scale_y = _item_scale(pruned, result, page_width, page_height)
        parsing = pruned.get("parsing_res_list")
        if isinstance(parsing, list):
            for parent_index, parent_value in enumerate(parsing):
                current_raw_index = raw_index
                raw_index += 1
                if not isinstance(parent_value, Mapping):
                    continue
                label = normalize_paddle_label(
                    parent_value.get("block_label", parent_value.get("label"))
                )
                text = str(parent_value.get("block_content") or "").strip()
                bbox = _record_bbox(
                    parent_value,
                    page_width=page_width,
                    page_height=page_height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    context=f"layoutParsingResults[{item_index}].prunedResult.parsing_res_list[{parent_index}]",
                )
                if (
                    bbox is not None
                    and text
                    and map_paddle_label_to_block_type(label) not in _NON_TEXT_PARENT_TYPES
                ):
                    parents.append(_ParentRecord(
                        raw_index=current_raw_index,
                        json_path=(
                            f"layoutParsingResults[{item_index}].prunedResult."
                            f"parsing_res_list[{parent_index}]"
                        ),
                        bbox=bbox,
                        text=text,
                    ))

        detection = pruned.get("layout_det_res")
        boxes = detection.get("boxes") if isinstance(detection, Mapping) else None
        if not isinstance(boxes, list):
            continue
        for detector_index, detector_value in enumerate(boxes):
            if not isinstance(detector_value, Mapping):
                continue
            label = normalize_paddle_label(
                detector_value.get("label", detector_value.get("block_label"))
            )
            if label != "inline_formula":
                continue
            context = (
                f"layoutParsingResults[{item_index}].prunedResult."
                f"layout_det_res.boxes[{detector_index}]"
            )
            bbox = _record_bbox(
                detector_value,
                page_width=page_width,
                page_height=page_height,
                scale_x=scale_x,
                scale_y=scale_y,
                context=context,
            )
            if bbox is None:
                raise PaddleInlineFormulaObservationError(
                    f"{context} has no non-empty inline-formula bbox"
                )
            observations.append(PaddleInlineFormulaDetectorObservation(
                item_index=item_index,
                detector_index=detector_index,
                bbox=bbox,
                score=_optional_float(detector_value.get("score")),
                raw_json_path=context,
            ))

    by_parent: dict[int, list[int]] = {}
    bound = list(observations)
    for observation_index, observation in enumerate(observations):
        containing = [parent for parent in parents if _contains(parent.bbox, observation.bbox)]
        if len(containing) != 1:
            continue
        parent = containing[0]
        bound[observation_index] = replace(
            observation,
            parent_raw_index=parent.raw_index,
            parent_json_path=parent.json_path,
            parent_text=parent.text,
        )
        by_parent.setdefault(parent.raw_index, []).append(observation_index)

    parent_by_index = {parent.raw_index: parent for parent in parents}
    for parent_raw_index, observation_indices in by_parent.items():
        parent = parent_by_index[parent_raw_index]
        spans = [match.group(0) for match in _FORMULA_SPAN_RE.finditer(parent.text)]
        ordered = _reading_order_indices(observation_indices, bound)
        if len(spans) != len(ordered):
            continue
        for span, observation_index in zip(spans, ordered):
            bound[observation_index] = replace(
                bound[observation_index],
                exact_parent_text=span,
            )
    return tuple(
        observation
        for observation in bound
        if not (
            observation.exact_parent_text
            and is_formula_marker_token(observation.exact_parent_text)
        )
    )


def _item_scale(
    pruned: Mapping[str, Any],
    result: Mapping[str, Any],
    page_width: int,
    page_height: int,
) -> tuple[float, float]:
    shape = result.get("dataInfo")
    if not isinstance(shape, Mapping):
        shape = pruned
    source_width = shape.get("width")
    source_height = shape.get("height")
    if (
        isinstance(source_width, (int, float))
        and not isinstance(source_width, bool)
        and source_width > 0
        and isinstance(source_height, (int, float))
        and not isinstance(source_height, bool)
        and source_height > 0
    ):
        return page_width / float(source_width), page_height / float(source_height)
    return 1.0, 1.0


def _record_bbox(
    record: Mapping[str, Any],
    *,
    page_width: int,
    page_height: int,
    scale_x: float,
    scale_y: float,
    context: str,
) -> BBox | None:
    value = next(
        (
            record[key]
            for key in (
                "block_bbox",
                "coordinate",
                "bbox",
                "block_polygon_points",
                "polygon_points",
                "points",
            )
            if key in record
        ),
        None,
    )
    coords = _xyxy(value)
    if coords is None:
        return None
    x1, y1, x2, y2 = coords
    bbox = BBox.from_xyxy(
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    ).clamp(page_width, page_height)
    if bbox.area <= 0:
        raise PaddleInlineFormulaObservationError(f"{context} has an empty bbox")
    return bbox


def _xyxy(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            x = _required_float(value["x"])
            y = _required_float(value["y"])
            return x, y, x + _required_float(value["w"]), y + _required_float(value["h"])
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return tuple(_required_float(value[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) == 4 and not all(isinstance(item, (list, tuple, Mapping)) for item in value):
        return tuple(_required_float(item) for item in value)  # type: ignore[return-value]
    points = [
        (_required_float(item[0]), _required_float(item[1]))
        for item in value
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2
    ]
    if len(points) < 2:
        return None
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _required_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaddleInlineFormulaObservationError(f"bbox coordinate must be numeric: {value!r}")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _required_float(value)


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


def _reading_order_indices(
    indices: list[int],
    observations: list[PaddleInlineFormulaDetectorObservation],
) -> list[int]:
    rows: list[list[int]] = []
    for index in sorted(
        indices,
        key=lambda value: (observations[value].bbox.y1, observations[value].bbox.x1),
    ):
        bbox = observations[index].bbox
        for row in rows:
            row_y1 = min(observations[value].bbox.y1 for value in row)
            row_y2 = max(observations[value].bbox.y2 for value in row)
            overlap = max(0, min(row_y2, bbox.y2) - max(row_y1, bbox.y1))
            if overlap / max(1, min(row_y2 - row_y1, bbox.h)) >= 0.25:
                row.append(index)
                break
        else:
            rows.append([index])
    rows.sort(key=lambda row: min(observations[value].bbox.y1 for value in row))
    return [
        index
        for row in rows
        for index in sorted(
            row,
            key=lambda value: (
                observations[value].bbox.x1,
                observations[value].bbox.y1,
                observations[value].detector_index,
            ),
        )
    ]


__all__ = [
    "PaddleInlineFormulaDetectorObservation",
    "PaddleInlineFormulaObservationError",
    "inline_formula_detector_observations",
]
