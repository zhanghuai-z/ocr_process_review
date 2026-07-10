"""Explicit typed-route fixture for native CharOCR unit tests.

The production runner accepts a page routing plan only.  These helpers keep
native-recognition tests focused on segmentation/recognition without reviving
the deprecated raw PP-VL line-route path.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import numpy as np

from app.models.charocr_routing import (
    BlockRoutingPlan,
    PageRoutingPlan,
    RoutingLine,
    RoutingPlan,
    RoutingSegment,
    TextSliceRoute,
)
from app.core.ocr_ir import is_cjk_char
from app.engines.hanwang import micro_recblock
from app.models import Page


XYXY = tuple[int, int, int, int]
_FORMULA_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


def run_micro_recblock_with_explicit_routes(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict[str, Any]],
    *,
    page_ocr_lines: list[Any] | None = None,
    **kwargs: Any,
):
    """Run the production native runner with a test-owned typed route map."""
    rows, routing_plan, page = build_explicit_native_route_fixture(
        image_bgr,
        ppvl_blocks,
        page_ocr_lines=page_ocr_lines,
    )
    return micro_recblock.run_micro_recblock(
        image_bgr,
        rows,
        routing_plan=routing_plan,
        page=page,
        **kwargs,
    )


def build_explicit_native_route_fixture(
    image_bgr: np.ndarray,
    ppvl_blocks: list[dict[str, Any]],
    *,
    page_ocr_lines: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], PageRoutingPlan, Page]:
    """Create a minimal explicit route contract from test-provided geometry.

    The fixture accepts block and optional line geometry declared by a test.
    It intentionally ignores legacy ``_layout_line_routes`` fields.  Formula
    subblocks are converted to typed formula segments because those fixtures
    express the structural condition under test.
    """
    height, width = image_bgr.shape[:2]
    page = Page(image_path="", width=width, height=height)
    rows = [dict(deepcopy(row)) for row in ppvl_blocks]
    plans: list[BlockRoutingPlan] = []
    hints = tuple(_line_hint(value) for value in (page_ocr_lines or ()))

    for block_index, row in enumerate(rows):
        uid = str(row.get(micro_recblock.ROUTE_ROW_LAYOUT_BLOCK_UID_KEY) or f"test-route-{block_index}")
        row[micro_recblock.ROUTE_ROW_LAYOUT_BLOCK_UID_KEY] = uid
        if not micro_recblock._is_text_label(micro_recblock._effective_label_for_block(row)):
            continue
        block_bbox = _bbox(row.get("block_bbox"), width, height)
        line_specs = [
            hint
            for hint in hints
            if _intersect(hint[1], block_bbox) is not None
        ]
        if not line_specs:
            line_specs = [(str(row.get("block_content") or ""), block_bbox)]
        formula_values = iter(_FORMULA_RE.findall(str(row.get("block_content") or "")))
        subblocks = _structural_subblocks(row, width, height, formula_values)
        lines = tuple(
            _routing_line(index, text, _intersect(line_bbox, block_bbox) or block_bbox, subblocks)
            for index, (text, line_bbox) in enumerate(line_specs)
            if _intersect(line_bbox, block_bbox) is not None
        )
        slices = tuple(
            TextSliceRoute(
                line_index=line.index,
                segment_index=segment_index,
                bbox=segment.bbox,
                carved=len(line.segments) > 1,
                kind=segment.kind,
            )
            for line in lines
            for segment_index, segment in enumerate(line.segments)
            if segment.kind.startswith("text_")
        )
        plans.append(BlockRoutingPlan(
            block_uid=uid,
            plan=RoutingPlan(lines=lines, text_slices=slices, has_layout_routes=bool(lines)),
        ))

    return rows, PageRoutingPlan(
        page_uid=page.uid,
        prepass_run_id="test-explicit-native-routes",
        blocks=tuple(plans),
    ), page


def _line_hint(value: Any) -> tuple[str, XYXY]:
    if isinstance(value, dict):
        return str(value.get("text") or ""), tuple(int(item) for item in value["bbox"])
    return str(getattr(value, "text", "") or ""), tuple(int(item) for item in getattr(value, "bbox"))


def _structural_subblocks(
    row: dict[str, Any],
    width: int,
    height: int,
    formula_values: Any,
) -> list[tuple[str, XYXY, str]]:
    values = row.get("_route_subblocks")
    if not isinstance(values, list):
        return []
    output: list[tuple[str, XYXY, str]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        label = str(value.get("block_label") or value.get("label") or "")
        bbox = _bbox(value.get("block_bbox"), width, height)
        text = str(value.get("block_content") or "")
        if not text and micro_recblock.is_formula_label(label):
            text = next(formula_values, "")
        kind = "formula" if micro_recblock.is_formula_label(label) else "skip"
        output.append((kind, bbox, text))
    return sorted(output, key=lambda item: (item[1][0], item[1][1], item[1][2], item[1][3]))


def _routing_line(
    index: int,
    text: str,
    line_bbox: XYXY,
    subblocks: list[tuple[str, XYXY, str]],
) -> RoutingLine:
    lx1, ly1, lx2, ly2 = line_bbox
    cuts = [
        (kind, overlap, bbox, formula_text)
        for kind, bbox, formula_text in subblocks
        if (overlap := _intersect(line_bbox, bbox)) is not None
    ]
    segments: list[RoutingSegment] = []
    cursor = lx1
    for kind, cut, content_bbox, formula_text in cuts:
        cx1, _cy1, cx2, _cy2 = cut
        if cursor < cx1:
            segments.append(RoutingSegment(kind=_text_kind(text), bbox=(cursor, ly1, cx1, ly2)))
        segments.append(RoutingSegment(
            kind=kind,
            bbox=cut,
            text=formula_text,
            content_bbox=content_bbox if kind == "formula" else None,
        ))
        cursor = max(cursor, cx2)
    if cursor < lx2:
        segments.append(RoutingSegment(kind=_text_kind(text), bbox=(cursor, ly1, lx2, ly2)))
    if not segments:
        segments.append(RoutingSegment(kind=_text_kind(text), bbox=line_bbox))
    return RoutingLine(index=index, bbox=line_bbox, segments=tuple(segments), source="test-explicit")


def _text_kind(text: str) -> str:
    value = str(text or "")
    has_latin_or_digit = any(char.isascii() and char.isalnum() for char in value)
    return "text_latin" if has_latin_or_digit and not any(is_cjk_char(char) for char in value) else "text_other"


def _bbox(value: Any, width: int, height: int) -> XYXY:
    x1, y1, x2, y2 = (int(item) for item in value)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
