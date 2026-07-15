"""Reconcile native bbox proposals against foreground pixel evidence.

The reconciler owns geometry only. It never reads or repairs recognized text.
When several native proposals claim one visual unit, it emits one token atom
instead of preserving multiple contradictory character boxes.
"""
from __future__ import annotations

from collections import defaultdict, deque
from statistics import median

import numpy as np

from app.geometry.foreground import ForegroundComponent, analyze_foreground_components
from app.models.char_geometry import (
    GeometryAtom,
    GeometryReconcileResult,
    NativeGeometryProposal,
    XYXY,
)


_MIN_COMPONENT_CLAIM = 0.05
_MIN_SECONDARY_COMPONENT_AREA_RATIO = 0.03
_SLOT_UNDERFILL_RATIO = 0.85
_SLOT_MAX_WIDTH_RATIO = 1.35
_SLOT_MAX_GAP_RATIO = 0.25
_STABLE_SLOT_CONFIDENCE = 0.35


def reconcile_char_geometry(
    image_bgr: np.ndarray,
    proposals: tuple[NativeGeometryProposal, ...],
    *,
    region_bbox: XYXY | None = None,
) -> GeometryReconcileResult:
    """Return geometry atoms backed by connected foreground components."""
    if not proposals:
        return GeometryReconcileResult(())

    ordered = tuple(sorted(proposals, key=lambda item: item.index))
    analysis = analyze_foreground_components(image_bgr, region_bbox)
    if not analysis.components:
        return GeometryReconcileResult(
            atoms=tuple(_native_atom(proposal) for proposal in ordered),
        )

    claims_by_component, components_by_proposal = _component_claims(
        ordered,
        analysis.components,
    )
    conflict_groups = _conflict_proposal_groups(
        claims_by_component,
        components_by_proposal,
        ordered,
        analysis.components,
    )
    conflict_groups = _extend_underfilled_slots(
        conflict_groups,
        ordered,
        components_by_proposal,
    )

    atoms: list[GeometryAtom] = []
    consumed: set[int] = set()
    proposals_by_index = {proposal.index: proposal for proposal in ordered}
    for indices in conflict_groups:
        if any(index in consumed for index in indices):
            continue
        group_proposals = tuple(proposals_by_index[index] for index in indices)
        component_indices = sorted({
            component_index
            for proposal in group_proposals
            for component_index in components_by_proposal.get(proposal.index, ())
        })
        kept_components = _drop_isolated_component_noise(
            tuple(analysis.components[index] for index in component_indices),
            tuple(component_indices),
        )
        if not kept_components:
            continue
        kept_component_indices, components = kept_components
        atoms.append(GeometryAtom(
            bbox=_union_bbox(tuple(component.bbox for component in components)),
            proposal_indices=indices,
            granularity="token",
            component_indices=kept_component_indices,
            reason="foreground_conflict_reconciled",
        ))
        consumed.update(indices)

    atoms.extend(
        _native_atom(proposal)
        for proposal in ordered
        if proposal.index not in consumed
    )
    atoms.sort(key=lambda atom: min(atom.proposal_indices))
    return GeometryReconcileResult(atoms=tuple(atoms))


def _component_claims(
    proposals: tuple[NativeGeometryProposal, ...],
    components: tuple[ForegroundComponent, ...],
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    proposal_claims: dict[int, list[int]] = defaultdict(list)
    component_claims: dict[int, list[int]] = defaultdict(list)
    for component_index, component in enumerate(components):
        component_area = _bbox_area(component.bbox)
        if component_area <= 0:
            continue
        for proposal in proposals:
            intersection = _intersection_area(component.bbox, proposal.bbox)
            if intersection / component_area < _MIN_COMPONENT_CLAIM:
                continue
            component_claims[component_index].append(proposal.index)
            proposal_claims[proposal.index].append(component_index)
    return (
        {key: tuple(value) for key, value in component_claims.items()},
        {key: tuple(value) for key, value in proposal_claims.items()},
    )


def _conflict_proposal_groups(
    claims_by_component: dict[int, tuple[int, ...]],
    components_by_proposal: dict[int, tuple[int, ...]],
    proposals: tuple[NativeGeometryProposal, ...],
    components: tuple[ForegroundComponent, ...],
) -> list[tuple[int, ...]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    proposals_by_index = {proposal.index: proposal for proposal in proposals}
    expected_slot_width = _expected_native_slot_width(proposals, [])
    dominant_components = {
        proposal.index: _dominant_component_index(
            proposal,
            components_by_proposal.get(proposal.index, ()),
            components,
        )
        for proposal in proposals
    }
    for component_index, proposal_indices in claims_by_component.items():
        if len(proposal_indices) < 2:
            continue
        for position, index in enumerate(proposal_indices):
            if dominant_components.get(index) != component_index:
                continue
            for other in proposal_indices[position + 1:]:
                left = proposals_by_index[index]
                right = proposals_by_index[other]
                if _intersection_area(left.bbox, right.bbox) <= 0:
                    continue
                union = _union_bbox((left.bbox, right.bbox))
                same_component = dominant_components.get(other) == component_index
                same_component_coherent = same_component and (
                    expected_slot_width <= 0
                    or _bbox_width(union) <= expected_slot_width * _SLOT_MAX_WIDTH_RATIO
                    or _is_horizontal_stroke(components[component_index].bbox)
                )
                underfilled_slot = (
                    expected_slot_width > 0
                    and max(left.confidence, right.confidence) <= _STABLE_SLOT_CONFIDENCE
                    and _bbox_width(union)
                    < expected_slot_width * _SLOT_UNDERFILL_RATIO
                )
                if not same_component_coherent and not underfilled_slot:
                    continue
                adjacency[index].add(other)
                adjacency[other].add(index)

    groups: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue = deque([start])
        group: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in group:
                continue
            group.add(current)
            queue.extend(adjacency[current] - group)
        visited.update(group)
        ordered = tuple(sorted(group))
        if _indices_are_contiguous(ordered):
            groups.append(ordered)
    return groups


def _dominant_component_index(
    proposal: NativeGeometryProposal,
    component_indices: tuple[int, ...],
    components: tuple[ForegroundComponent, ...],
) -> int | None:
    if not component_indices:
        return None
    proposal_center = _bbox_center(proposal.bbox)
    largest_area = max(components[index].area for index in component_indices)
    centered = tuple(
        index
        for index in component_indices
        if _point_in_bbox(_bbox_center(components[index].bbox), proposal.bbox)
        and components[index].area >= largest_area * _MIN_SECONDARY_COMPONENT_AREA_RATIO
    )
    if centered:
        return min(
            centered,
            key=lambda index: _center_distance_squared(
                proposal_center,
                _bbox_center(components[index].bbox),
            ),
        )
    return max(
        component_indices,
        key=lambda index: (
            _intersection_area(proposal.bbox, components[index].bbox),
            components[index].area,
        ),
    )


def _bbox_center(bbox: XYXY) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _point_in_bbox(point: tuple[float, float], bbox: XYXY) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _center_distance_squared(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _extend_underfilled_slots(
    groups: list[tuple[int, ...]],
    proposals: tuple[NativeGeometryProposal, ...],
    components_by_proposal: dict[int, tuple[int, ...]],
) -> list[tuple[int, ...]]:
    expected_width = _expected_native_slot_width(proposals, groups)
    if expected_width <= 0:
        return groups
    by_index = {proposal.index: proposal for proposal in proposals}
    occupied = {index for group in groups for index in group}
    result: list[tuple[int, ...]] = []
    for group in groups:
        current = list(group)
        bbox = _union_bbox(tuple(by_index[index].bbox for index in current))
        while _bbox_width(bbox) < expected_width * _SLOT_UNDERFILL_RATIO:
            next_index = current[-1] + 1
            candidate = by_index.get(next_index)
            if candidate is None or next_index in occupied:
                break
            gap = max(0, candidate.bbox[0] - bbox[2])
            extended = _union_bbox((bbox, candidate.bbox))
            if gap > expected_width * _SLOT_MAX_GAP_RATIO:
                break
            if _bbox_width(extended) > expected_width * _SLOT_MAX_WIDTH_RATIO:
                break
            if _vertical_overlap_fraction(bbox, candidate.bbox) < 0.50:
                break
            if not components_by_proposal.get(candidate.index):
                break
            current.append(candidate.index)
            occupied.add(candidate.index)
            bbox = extended
        result.append(tuple(current))
    return result


def _expected_native_slot_width(
    proposals: tuple[NativeGeometryProposal, ...],
    groups: list[tuple[int, ...]],
) -> float:
    conflicted = {index for group in groups for index in group}
    widths = [
        _bbox_width(proposal.bbox)
        for proposal in proposals
        if proposal.index not in conflicted
        and proposal.confidence > _STABLE_SLOT_CONFIDENCE
        and 0.55 <= _bbox_width(proposal.bbox) / max(1, _bbox_height(proposal.bbox)) <= 1.60
    ]
    if widths:
        return float(median(widths))
    fallback_widths = sorted(
        _bbox_width(proposal.bbox)
        for proposal in proposals
        if 0.45 <= _bbox_width(proposal.bbox) / max(1, _bbox_height(proposal.bbox)) <= 1.80
    )
    if not fallback_widths:
        return 0.0
    upper_half = fallback_widths[len(fallback_widths) // 2:]
    return float(median(upper_half))


def _is_horizontal_stroke(bbox: XYXY) -> bool:
    return _bbox_width(bbox) >= max(4, _bbox_height(bbox) * 4)


def _drop_isolated_component_noise(
    components: tuple[ForegroundComponent, ...],
    component_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[ForegroundComponent, ...]] | None:
    if not components:
        return None
    largest = max(component.area for component in components)
    kept = tuple(
        (index, component)
        for index, component in zip(component_indices, components)
        if component.area >= max(3, largest * _MIN_SECONDARY_COMPONENT_AREA_RATIO)
    )
    if not kept:
        return None
    return tuple(item[0] for item in kept), tuple(item[1] for item in kept)


def _native_atom(proposal: NativeGeometryProposal) -> GeometryAtom:
    return GeometryAtom(
        bbox=proposal.bbox,
        proposal_indices=(proposal.index,),
    )


def _indices_are_contiguous(indices: tuple[int, ...]) -> bool:
    return indices == tuple(range(indices[0], indices[-1] + 1))


def _bbox_area(bbox: XYXY) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_width(bbox: XYXY) -> int:
    return max(0, bbox[2] - bbox[0])


def _bbox_height(bbox: XYXY) -> int:
    return max(0, bbox[3] - bbox[1])


def _intersection_area(left: XYXY, right: XYXY) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _vertical_overlap_fraction(left: XYXY, right: XYXY) -> float:
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    smaller = min(_bbox_height(left), _bbox_height(right))
    return overlap / smaller if smaller else 0.0


def _union_bbox(boxes: tuple[XYXY, ...]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


__all__ = ["reconcile_char_geometry"]
