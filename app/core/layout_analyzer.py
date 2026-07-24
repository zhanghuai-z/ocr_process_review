"""Pure Paddle layout response normalization.

This module owns only the conversion from the vendor response contract to an
immutable :class:`LayoutSnapshot`.  It has no filesystem, HTTP, UI, or
project-session responsibilities.  The application service owns artifact
creation and repository adoption.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.adapters.paddle.inline_formula_observations import (
    PaddleInlineFormulaObservationError,
    inline_formula_detector_observations,
)
from app.core.paddle_labels import normalize_paddle_label
from app.models.entity_id import new_entity_uid
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot


class LayoutAnalysisError(ValueError):
    """The Paddle response cannot produce a valid layout snapshot."""


_LABEL_TO_BLOCK_TYPE: dict[str, BlockType] = {
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "paragraph_text": BlockType.TEXT,
    "plain_text": BlockType.TEXT,
    "body": BlockType.TEXT,
    "body_text": BlockType.TEXT,
    "content": BlockType.TEXT,
    "doc_text": BlockType.TEXT,
    "text_region": BlockType.TEXT,
    "text_box": BlockType.TEXT,
    "header": BlockType.TEXT,
    "footer": BlockType.TEXT,
    "footnote": BlockType.TEXT,
    "number": BlockType.TEXT,
    "page_number": BlockType.TEXT,
    "reference": BlockType.REFERENCE,
    "reference_content": BlockType.REFERENCE,
    "references": BlockType.REFERENCE,
    "reference_list": BlockType.REFERENCE,
    "bibliography": BlockType.REFERENCE,
    "title": BlockType.TITLE,
    "doc_title": BlockType.TITLE,
    "doc_heading": BlockType.TITLE,
    "section_title": BlockType.TITLE,
    "chapter_title": BlockType.TITLE,
    "paragraph_title": BlockType.TITLE,
    "text_title": BlockType.TITLE,
    "heading": BlockType.TITLE,
    "heading_1": BlockType.TITLE,
    "heading_2": BlockType.TITLE,
    "heading_3": BlockType.TITLE,
    "heading_4": BlockType.TITLE,
    "heading_5": BlockType.TITLE,
    "heading_6": BlockType.TITLE,
    "figure": BlockType.FIGURE,
    "graphic": BlockType.FIGURE,
    "photo": BlockType.FIGURE,
    "image": BlockType.FIGURE,
    "picture": BlockType.FIGURE,
    "illustration": BlockType.FIGURE,
    "chart": BlockType.FIGURE,
    "logo": BlockType.FIGURE,
    "seal": BlockType.FIGURE,
    "figure_caption": BlockType.FIGURE_CAPTION,
    "figure_title": BlockType.FIGURE_CAPTION,
    "caption": BlockType.FIGURE_CAPTION,
    "image_caption": BlockType.FIGURE_CAPTION,
    "table": BlockType.TABLE,
    "table_region": BlockType.TABLE,
    "table_block": BlockType.TABLE,
    "table_body": BlockType.TABLE,
    "table_cell": BlockType.TABLE,
    "table_caption": BlockType.TABLE_CAPTION,
    "table_caption_text": BlockType.TABLE_CAPTION,
    "table_note": BlockType.TABLE_CAPTION,
    "table_title": BlockType.TABLE_CAPTION,
    "equation": BlockType.EQUATION,
    "equation_block": BlockType.EQUATION,
    "display_formula": BlockType.EQUATION,
    "isolated_formula": BlockType.EQUATION,
    "inline_formula": BlockType.EQUATION,
    "formula": BlockType.EQUATION,
    "formula_block": BlockType.EQUATION,
    "formula_number": BlockType.EQUATION,
    "math": BlockType.EQUATION,
    "math_formula": BlockType.EQUATION,
    "math_block": BlockType.EQUATION,
}


def _block_type(label: str) -> BlockType:
    return _LABEL_TO_BLOCK_TYPE.get(label, BlockType.UNKNOWN)


def _ocr_policy(block_type: BlockType) -> OcrPolicy:
    if block_type is BlockType.EQUATION:
        return OcrPolicy.PRESERVE_AS_FORMULA
    if block_type is BlockType.TABLE:
        return OcrPolicy.PRESERVE_AS_TABLE
    if block_type in {BlockType.FIGURE, BlockType.UNKNOWN}:
        return OcrPolicy.SKIP
    return OcrPolicy.TEXT_OCR


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayoutAnalysisError(f"{context} must be an object")
    return value


def _positive_dimension(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LayoutAnalysisError(f"{field_name} must be a positive integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayoutAnalysisError(f"{context} must be numeric")
    return float(value)


def _xyxy(value: object, context: str) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            x = _number(value["x"], f"{context}.x")
            y = _number(value["y"], f"{context}.y")
            w = _number(value["w"], f"{context}.w")
            h = _number(value["h"], f"{context}.h")
            return x, y, x + w, y + h
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return tuple(
                _number(value[key], f"{context}.{key}")
                for key in ("x1", "y1", "x2", "y2")
            )  # type: ignore[return-value]
        raise LayoutAnalysisError(f"{context} has no supported coordinate fields")

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LayoutAnalysisError(f"{context} must be coordinates or points")
    if len(value) == 4 and not all(
        isinstance(item, (list, tuple, Mapping)) for item in value
    ):
        return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]

    points: list[tuple[float, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            raise LayoutAnalysisError(f"{context}[{index}] must be a point")
        points.append(
            (
                _number(item[0], f"{context}[{index}][0]"),
                _number(item[1], f"{context}[{index}][1]"),
            )
        )
    if len(points) < 2:
        raise LayoutAnalysisError(f"{context} must contain at least two points")
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _record_bbox(record: Mapping[str, Any], *, width: int, height: int, scale_x: float, scale_y: float, context: str) -> BBox:
    coordinate_key = next(
        (
            key
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
    if coordinate_key is None:
        raise LayoutAnalysisError(f"{context} has no layout bbox")
    x1, y1, x2, y2 = _xyxy(record[coordinate_key], f"{context}.{coordinate_key}")
    bbox = BBox.from_xyxy(
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    ).clamp(width, height)
    if bbox.w <= 0 or bbox.h <= 0:
        raise LayoutAnalysisError(f"{context} has an empty layout bbox")
    return bbox


def _label(record: Mapping[str, Any], context: str) -> tuple[str, str]:
    raw = record.get("block_label", record.get("label"))
    if not isinstance(raw, str) or not raw.strip():
        raise LayoutAnalysisError(f"{context} has no layout label")
    vendor_label = raw.strip()
    return vendor_label, normalize_paddle_label(vendor_label)


def _score(record: Mapping[str, Any], context: str) -> float | None:
    for key in ("score", "confidence", "block_score"):
        if key not in record or record[key] is None:
            continue
        return _number(record[key], f"{context}.{key}")
    return None


def _scale_for_item(item: Mapping[str, Any], result: Mapping[str, Any], width: int, height: int) -> tuple[float, float]:
    shape: Mapping[str, Any] | None = None
    data_info = result.get("dataInfo")
    if isinstance(data_info, Mapping):
        shape = data_info
    else:
        pruned = item.get("prunedResult")
        if isinstance(pruned, Mapping) and isinstance(pruned.get("width"), (int, float)) and isinstance(pruned.get("height"), (int, float)):
            shape = pruned
    if shape is None:
        return 1.0, 1.0
    source_width = shape.get("width")
    source_height = shape.get("height")
    if isinstance(source_width, bool) or isinstance(source_height, bool):
        raise LayoutAnalysisError("Paddle response dimensions must be numeric")
    if not isinstance(source_width, (int, float)) or not isinstance(source_height, (int, float)):
        raise LayoutAnalysisError("Paddle response dimensions must be numeric")
    if source_width <= 0 or source_height <= 0:
        raise LayoutAnalysisError("Paddle response dimensions must be positive")
    return width / float(source_width), height / float(source_height)


def _records_for_item(item: Mapping[str, Any], context: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    pruned = _mapping(item.get("prunedResult"), f"{context}.prunedResult")
    parsing = pruned.get("parsing_res_list")
    if parsing is not None:
        if not isinstance(parsing, list):
            raise LayoutAnalysisError(f"{context}.prunedResult.parsing_res_list must be an array")
        if parsing:
            return tuple(
                (f"{context}.prunedResult.parsing_res_list[{index}]", _mapping(value, f"{context}.parsing_res_list[{index}]"))
                for index, value in enumerate(parsing)
            )

    detection = pruned.get("layout_det_res")
    if isinstance(detection, Mapping):
        boxes = detection.get("boxes")
        if boxes is not None:
            if not isinstance(boxes, list):
                raise LayoutAnalysisError(f"{context}.prunedResult.layout_det_res.boxes must be an array")
            return tuple(
                (f"{context}.prunedResult.layout_det_res.boxes[{index}]", _mapping(value, f"{context}.layout_det_res.boxes[{index}]"))
                for index, value in enumerate(boxes)
            )
    if parsing is None:
        raise LayoutAnalysisError(f"{context}.prunedResult.parsing_res_list is missing")
    return ()


class LayoutAnalyzer:
    """Normalize one successful Paddle VL response into layout truth."""

    def analyze(
        self,
        response: Mapping[str, Any],
        *,
        page_uid: str,
        page_width: int,
        page_height: int,
        artifact_uid: str,
        source_run_id: str,
        source_engine: str = "paddleocr-vl-1.6",
        revision: int = 1,
    ) -> LayoutSnapshot:
        if not isinstance(response, Mapping):
            raise LayoutAnalysisError("Paddle response must be an object")
        if not isinstance(page_uid, str) or not page_uid.strip():
            raise LayoutAnalysisError("page_uid must be non-empty")
        if not isinstance(artifact_uid, str) or not artifact_uid.strip():
            raise LayoutAnalysisError("artifact_uid must be non-empty")
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise LayoutAnalysisError("source_run_id must be non-empty")
        if not isinstance(source_engine, str) or not source_engine.strip():
            raise LayoutAnalysisError("source_engine must be non-empty")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise LayoutAnalysisError("revision must be a positive integer")
        width = _positive_dimension(page_width, "page_width")
        height = _positive_dimension(page_height, "page_height")

        error_code = response.get("errorCode")
        if error_code not in (None, 0, "0"):
            raise LayoutAnalysisError(
                f"Paddle response failed: {response.get('errorMsg') or error_code}"
            )
        error_message = response.get("errorMsg")
        if error_message not in (None, "", "Success"):
            raise LayoutAnalysisError(f"Paddle response failed: {error_message}")

        result = _mapping(response.get("result"), "Paddle response.result")
        items = result.get("layoutParsingResults")
        if not isinstance(items, list):
            raise LayoutAnalysisError("Paddle response.result.layoutParsingResults must be an array")

        blocks: list[LayoutBlockSnapshot] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        raw_index = 0
        for item_index, raw_item in enumerate(items):
            item = _mapping(raw_item, f"layoutParsingResults[{item_index}]")
            scale_x, scale_y = _scale_for_item(item, result, width, height)
            for record_path, record in _records_for_item(item, f"layoutParsingResults[{item_index}]"):
                current_raw_index = raw_index
                raw_index += 1
                vendor_label, normalized_label = _label(record, record_path)
                bbox = _record_bbox(
                    record,
                    width=width,
                    height=height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    context=record_path,
                )
                signature = (normalized_label, bbox.to_xyxy())
                if signature in seen:
                    continue
                seen.add(signature)
                block_type = _block_type(normalized_label)
                origin = BlockOrigin(
                    created_by=BlockSource.AUTO_LAYOUT.value,
                    source_engine=source_engine,
                    source_run_id=source_run_id,
                    vendor_label=vendor_label,
                    source_confidence=_score(record, record_path),
                    original_bbox=bbox,
                    original_kind=block_type,
                    raw_artifact_uid=artifact_uid,
                    raw_json_path=record_path,
                    raw_index=current_raw_index,
                )
                blocks.append(
                    LayoutBlockSnapshot(
                        block_type=block_type,
                        bbox=bbox,
                        order=len(blocks),
                        source_label=normalized_label,
                        origin=origin,
                        ocr_policy=_ocr_policy(block_type),
                        authorship=BlockSource.AUTO_LAYOUT,
                        uid=new_entity_uid("block"),
                    )
                )

        try:
            inline_formulas = inline_formula_detector_observations(
                response,
                page_width=width,
                page_height=height,
            )
        except PaddleInlineFormulaObservationError as exc:
            raise LayoutAnalysisError(str(exc)) from exc
        equation_blocks = [
            block for block in blocks if block.block_type is BlockType.EQUATION
        ]
        for observation in inline_formulas:
            if any(_bbox_contains(block.bbox, observation.bbox) for block in equation_blocks):
                continue
            origin = BlockOrigin(
                created_by=BlockSource.AUTO_LAYOUT.value,
                source_engine=source_engine,
                source_run_id=source_run_id,
                vendor_label="inline_formula",
                source_confidence=observation.score,
                original_bbox=observation.bbox,
                original_kind=BlockType.EQUATION,
                raw_artifact_uid=artifact_uid,
                raw_json_path=observation.raw_json_path,
                raw_index=None,
            )
            block = LayoutBlockSnapshot(
                block_type=BlockType.EQUATION,
                bbox=observation.bbox,
                order=len(blocks),
                source_label="inline_formula",
                origin=origin,
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                authorship=BlockSource.AUTO_LAYOUT,
                uid=new_entity_uid("block"),
            )
            blocks.append(block)
            equation_blocks.append(block)

        return LayoutSnapshot(
            page_uid=page_uid,
            revision=revision,
            artifact_uid=artifact_uid,
            source_engine=source_engine,
            source_run_id=source_run_id,
            blocks=tuple(blocks),
        )


def _bbox_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


__all__ = ["LayoutAnalysisError", "LayoutAnalyzer"]
