"""Derived state for user-edited Paddle inline formula anchors."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.core.bbox_extraction import bbox_from_variant
from app.core.paddle_labels import normalize_paddle_label
from app.models import Block, LayoutEditEvent, Page


HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY = "handled_inline_formula_origin_bbox"


def inline_formula_origin_bbox(block: Block) -> tuple[int, int, int, int] | None:
    origin = block.origin
    if origin is None or origin.original_bbox is None:
        return None
    if normalize_paddle_label(origin.source_label) != "inline_formula":
        return None
    return origin.original_bbox.to_xyxy()


def handled_inline_formula_origin_bboxes(page: Page) -> set[tuple[int, int, int, int]]:
    handled: set[tuple[int, int, int, int]] = set()
    for event in page.layout_edit_events:
        bbox = _event_handled_origin_bbox(event)
        if bbox is not None:
            handled.add(bbox)
    return handled


def mark_inline_formula_origin_handled(page: Page, block: Block, *, op: str) -> bool:
    origin = inline_formula_origin_bbox(block)
    if origin is None:
        return False
    if origin in handled_inline_formula_origin_bboxes(page):
        return False
    page.layout_edit_events.append(
        LayoutEditEvent(
            page_uid=page.uid,
            target_uid=block.uid,
            op=op,
            before={"origin_bbox": list(origin)},
            after={HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY: list(origin)},
            actor="user",
        )
    )
    return True


def filter_handled_inline_formula_subblocks(
    page: Page,
    values: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    handled = handled_inline_formula_origin_bboxes(page)
    if not handled:
        return [dict(value) for value in values if isinstance(value, dict)]
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if _inline_formula_subblock_bbox(page, value) in handled:
            continue
        result.append(dict(value))
    return result


def _event_handled_origin_bbox(event: LayoutEditEvent) -> tuple[int, int, int, int] | None:
    payload = event.after.get(HANDLED_INLINE_FORMULA_ORIGIN_BBOX_KEY)
    return _xyxy_tuple(payload)


def _inline_formula_subblock_bbox(page: Page, value: dict[str, Any]) -> tuple[int, int, int, int] | None:
    label = str(value.get("block_label") or value.get("label") or value.get("type") or "")
    if normalize_paddle_label(label) != "inline_formula":
        return None
    bbox = bbox_from_variant(
        value.get("block_bbox") or value.get("bbox") or value.get("coordinate"),
        max_w=page.width,
        max_h=page.height,
    )
    if bbox is None or bbox.area <= 0:
        return None
    return bbox.clamp(page.width, page.height).to_xyxy()


def _xyxy_tuple(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(int(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None
