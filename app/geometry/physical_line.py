"""Resolve PP line proposals into uniquely owned physical text rows.

This module owns geometry only. Layout supplies ownership context and structural
exclusions; OCR adapters retain text and vendor metadata. The resolver never
mutates either source.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.geometry.foreground import ForegroundComponent, analyze_foreground_components
from app.models.physical_line_geometry import (
    LineGeometryContext,
    LineGeometrySeed,
    PhysicalLineResolution,
    ResolvedPhysicalLine,
    TEXT_AXIS_HORIZONTAL,
    TEXT_AXIS_VERTICAL,
    XYXY,
)


class PhysicalLineGeometryResolver:
    """Assign foreground components and derive one bbox per physical row."""

    def resolve(
        self,
        contexts: tuple[LineGeometryContext, ...],
        page_image_bgr: np.ndarray | None,
    ) -> PhysicalLineResolution:
        analysis = analyze_foreground_components(
            page_image_bgr,
            detect_polarity=False,
        )
        page_bbox = analysis.region_bbox
        claimed_components: set[int] = set()
        resolved: list[ResolvedPhysicalLine] = []
        for context in contexts:
            merged_rows = _merge_co_baseline_seeds(context.block_uid, context.seeds)
            for row in merged_rows:
                component_indices, bbox = _resolve_row_components(
                    row,
                    merged_rows,
                    context,
                    analysis.components,
                    page_bbox,
                    claimed_components,
                )
                claimed_components.update(component_indices)
                resolved.append(replace(
                    row,
                    bbox=bbox,
                    component_indices=component_indices,
                ))
        return PhysicalLineResolution(tuple(resolved))


def resolve_physical_lines(
    contexts: tuple[LineGeometryContext, ...],
    page_image_bgr: np.ndarray | None,
) -> PhysicalLineResolution:
    return PhysicalLineGeometryResolver().resolve(contexts, page_image_bgr)


def _merge_co_baseline_seeds(
    owner_block_uid: str,
    seeds: tuple[LineGeometrySeed, ...],
) -> tuple[ResolvedPhysicalLine, ...]:
    pending = list(sorted(seeds, key=_row_sort_key))
    rows: list[ResolvedPhysicalLine] = []
    while pending:
        seed = pending.pop(0)
        members = [seed]
        changed = True
        while changed:
            changed = False
            for candidate in pending[:]:
                if any(_same_physical_row(candidate, member) for member in members):
                    members.append(candidate)
                    pending.remove(candidate)
                    changed = True
        ordered = sorted(members, key=_primary_sort_key)
        rows.append(ResolvedPhysicalLine(
            owner_block_uid=owner_block_uid,
            representative_index=min(item.source_index for item in ordered),
            source_indices=tuple(item.source_index for item in ordered),
            bbox=_union_bbox(tuple(item.bbox for item in ordered)),
            text_axis=ordered[0].text_axis,
            orientation_angle=ordered[0].orientation_angle,
            has_word_geometry=any(item.has_word_geometry for item in ordered),
        ))
    return tuple(rows)


def _resolve_row_components(
    row: ResolvedPhysicalLine,
    peer_rows: tuple[ResolvedPhysicalLine, ...],
    context: LineGeometryContext,
    components: tuple[ForegroundComponent, ...],
    page_bbox: XYXY,
    claimed_components: set[int],
) -> tuple[tuple[int, ...], XYXY]:
    if not components:
        return (), row.bbox

    accepted: list[int] = []
    accepted_boxes: list[XYXY] = []
    active_bbox = row.bbox
    primary_domain = _union_span(
        _primary_span(row.bbox, row.text_axis),
        _primary_span(context.block_bbox, row.text_axis),
    )
    changed = True
    while changed:
        changed = False
        active_cross = _cross_span(active_bbox, row.text_axis)
        for component_index, component in enumerate(components):
            if component_index in claimed_components or component_index in accepted:
                continue
            center_x, center_y = _center(component.bbox)
            primary = center_x if row.text_axis == TEXT_AXIS_HORIZONTAL else center_y
            cross = center_y if row.text_axis == TEXT_AXIS_HORIZONTAL else center_x
            if not (primary_domain[0] <= primary <= primary_domain[1]):
                continue
            if not (active_cross[0] <= cross <= active_cross[1]):
                continue
            if any(_contains_point(bbox, center_x, center_y) for bbox in context.excluded_bboxes):
                continue
            if any(
                peer.representative_index != row.representative_index
                and _contains_point(peer.bbox, center_x, center_y)
                for peer in peer_rows
            ):
                continue
            if _crosses_another_row_center(component, row, peer_rows):
                continue
            if _spans_page(component.bbox, page_bbox):
                continue
            accepted.append(component_index)
            accepted_boxes.append(component.bbox)
            active_bbox = _union_bbox((active_bbox, component.bbox))
            changed = True

    if not accepted_boxes:
        return (), row.bbox
    owned_bbox = _union_bbox(tuple(accepted_boxes))
    if row.text_axis == TEXT_AXIS_HORIZONTAL:
        bbox = (
            min(row.bbox[0], owned_bbox[0]),
            owned_bbox[1] if row.has_word_geometry else row.bbox[1],
            max(row.bbox[2], owned_bbox[2]),
            owned_bbox[3] if row.has_word_geometry else row.bbox[3],
        )
    else:
        bbox = (
            owned_bbox[0] if row.has_word_geometry else row.bbox[0],
            min(row.bbox[1], owned_bbox[1]),
            owned_bbox[2] if row.has_word_geometry else row.bbox[2],
            max(row.bbox[3], owned_bbox[3]),
        )
    return tuple(accepted), bbox


def _same_physical_row(left: LineGeometrySeed, right: LineGeometrySeed) -> bool:
    if left.text_axis != right.text_axis or left.orientation_angle != right.orientation_angle:
        return False
    left_cross = _cross_span(left.bbox, left.text_axis)
    right_cross = _cross_span(right.bbox, right.text_axis)
    left_center = sum(left_cross) / 2.0
    right_center = sum(right_cross) / 2.0
    return (
        right_cross[0] <= left_center <= right_cross[1]
        and left_cross[0] <= right_center <= left_cross[1]
    )


def _row_sort_key(seed: LineGeometrySeed) -> tuple[int, int, int, int]:
    cross = _cross_span(seed.bbox, seed.text_axis)
    primary = _primary_span(seed.bbox, seed.text_axis)
    return (
        0 if seed.text_axis == TEXT_AXIS_HORIZONTAL else 1,
        cross[0],
        primary[0],
        seed.source_index,
    )


def _primary_sort_key(seed: LineGeometrySeed) -> tuple[int, int]:
    start = _primary_span(seed.bbox, seed.text_axis)[0]
    if seed.orientation_angle == 180:
        start = -start
    return start, seed.source_index


def _primary_span(bbox: XYXY, axis: str) -> tuple[int, int]:
    return (bbox[0], bbox[2]) if axis == TEXT_AXIS_HORIZONTAL else (bbox[1], bbox[3])


def _cross_span(bbox: XYXY, axis: str) -> tuple[int, int]:
    return (bbox[1], bbox[3]) if axis == TEXT_AXIS_HORIZONTAL else (bbox[0], bbox[2])


def _union_span(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return min(left[0], right[0]), max(left[1], right[1])


def _center(bbox: XYXY) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _contains_point(bbox: XYXY, x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _spans_page(inner: XYXY, outer: XYXY) -> bool:
    return (
        (inner[0] <= outer[0] and inner[2] >= outer[2])
        or (inner[1] <= outer[1] and inner[3] >= outer[3])
    )


def _crosses_another_row_center(
    component: ForegroundComponent,
    row: ResolvedPhysicalLine,
    peer_rows: tuple[ResolvedPhysicalLine, ...],
) -> bool:
    component_cross = _cross_span(component.bbox, row.text_axis)
    return any(
        peer.representative_index != row.representative_index
        and component_cross[0]
        <= sum(_cross_span(peer.bbox, peer.text_axis)) / 2.0
        <= component_cross[1]
        for peer in peer_rows
        if peer.text_axis == row.text_axis
    )


def _union_bbox(boxes: tuple[XYXY, ...]) -> XYXY:
    return (
        min(bbox[0] for bbox in boxes),
        min(bbox[1] for bbox in boxes),
        max(bbox[2] for bbox in boxes),
        max(bbox[3] for bbox in boxes),
    )


__all__ = ["PhysicalLineGeometryResolver", "resolve_physical_lines"]
