"""Build CharOCR text crops from PP-OCRv6 line observations.

PP-OCR word boxes are geometry proposals for Latin/digit routing. For every
line that contains Latin letters or digits, Latin/digit proposals create
EngCut masks, isolated punctuation becomes an explicit PP-OCR observation,
and every remaining horizontal region is dispatched to LineCut.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from unicodedata import category

import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.models.charocr_routing import (
    COMPONENT_GROUPING_SINGLE_GLYPH,
    PpOcrLatinTokenObservation,
    RoutingSegment,
)
from app.core.ocr_ir import is_cjk_char


XYXY = tuple[int, int, int, int]
_MIN_COMPONENT_AREA = 3


@dataclass(frozen=True)
class RoutePartitionIssue:
    code: str
    message: str
    bbox: XYXY


@dataclass(frozen=True)
class RoutePartition:
    segments: tuple[RoutingSegment, ...]
    issues: tuple[RoutePartitionIssue, ...] = ()


def partition_charocr_text_region(
    image_bgr: np.ndarray | None,
    prepass_line: PpOcrV6LineHint,
    region_bbox: XYXY,
) -> RoutePartition:
    """Partition one non-structural PP-OCR row region into native OCR crops.

    CJK-only rows remain one LineCut route.  On a Latin/digit row, PP-OCR
    Latin word boxes define EngCut masks.  Component closure may repair a
    word-box edge, but ink anchored by any other PP-OCR token cannot enter a
    Latin mask.  Everything outside those masks remains a LineCut region.
    """
    text = str(prepass_line.text or "")
    if not _has_latin_or_digit(text):
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox, text=text),))
    if not prepass_line.words:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_missing_word_boxes",
                message="PP-OCRv6 Latin/digit line has no word-box proposals",
                bbox=region_bbox,
            ),
        ))
    if image_bgr is None:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_requires_page_image",
                message="PP-OCRv6 Latin/digit routing requires the page image",
                bbox=region_bbox,
            ),
        ))

    components = _ink_components(image_bgr, region_bbox)
    if not components:
        return RoutePartition(())

    tokens = tuple(
        token
        for token in prepass_line.words
        if str(token.text or "").strip() and _intersect(token.bbox, region_bbox) is not None
    )
    if not tokens:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="latin_line_no_tokens_in_text_region",
                message="PP-OCRv6 Latin/digit line has no word boxes in its text region",
                bbox=region_bbox,
            ),
        ))

    mixed_tokens = [
        token
        for token in tokens
        if _has_latin_or_digit(token.text) and any(is_cjk_char(char) for char in str(token.text or ""))
    ]
    if mixed_tokens:
        token = mixed_tokens[0]
        return RoutePartition((), (
            RoutePartitionIssue(
                code="mixed_word_token",
                message=f"PP-OCRv6 mixed CJK/Latin token cannot form a deterministic mask: {token.text!r}",
                bbox=_clip(token.bbox, region_bbox),
            ),
        ))

    latin_tokens = tuple(token for token in tokens if _token_branch(token.text) == "latin")
    if not latin_tokens:
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox),))

    component_owners = _component_owner_token_indices(components, tokens, region_bbox)
    masks: list[tuple[PpOcrV6WordBox, XYXY]] = []
    issues: list[RoutePartitionIssue] = []
    for token in latin_tokens:
        mask_bbox = _latin_mask_bbox(
            components,
            token,
            component_owners,
            region_bbox,
        )
        if mask_bbox is None:
            if _clip(token.bbox, region_bbox) != token.bbox:
                # A structural cut owns the rest of this PP proposal.  If the
                # text-side fragment has no ink, it must not become an EngCut
                # requirement at the formula/table boundary.
                continue
            issues.append(RoutePartitionIssue(
                code="missing_latin_token_ink",
                message=f"PP-OCRv6 Latin/digit token has no unambiguous ink: {token.text!r}",
                bbox=_clip(token.bbox, region_bbox),
            ))
            continue
        masks.append((token, mask_bbox))

    if issues:
        return RoutePartition((), tuple(issues))
    return _segments_from_latin_masks(
        region_bbox,
        masks,
        components,
        tokens=tokens,
        component_owners=component_owners,
    )


def _segments_from_latin_masks(
    region_bbox: XYXY,
    masks: list[tuple[PpOcrV6WordBox, XYXY]],
    components: list[tuple[int, int, int, int, int]],
    *,
    tokens: tuple[PpOcrV6WordBox, ...],
    component_owners: dict[tuple[int, int, int, int, int], int | None],
) -> RoutePartition:
    groups: list[tuple[list[tuple[PpOcrV6WordBox, XYXY]], XYXY]] = []
    current_tokens: list[tuple[PpOcrV6WordBox, XYXY]] = []
    current_bbox: XYXY | None = None
    rx1, ry1, rx2, ry2 = region_bbox
    for token, bbox in sorted(masks, key=lambda item: (item[1][0], item[0].token_index)):
        if current_bbox is None:
            current_tokens = [(token, bbox)]
            current_bbox = bbox
            continue
        if bbox[0] < current_bbox[2]:
            return RoutePartition((), (
                RoutePartitionIssue(
                    code="overlapping_latin_masks",
                    message="PP-OCRv6 Latin/digit masks overlap after component closure",
                    bbox=(bbox[0], ry1, current_bbox[2], ry2),
                ),
            ))
        gap = (current_bbox[2], ry1, bbox[0], ry2)
        if _has_visible_ink(components, gap):
            groups.append((current_tokens, current_bbox))
            current_tokens = [(token, bbox)]
            current_bbox = bbox
            continue
        current_tokens.append((token, bbox))
        current_bbox = _union([current_bbox, bbox])
    if current_bbox is not None:
        groups.append((current_tokens, current_bbox))

    segments: list[RoutingSegment] = []
    cursor = rx1
    for group_items, bbox in groups:
        group_tokens = [token for token, _token_bbox in group_items]
        x1, _y1, x2, _y2 = bbox
        before = (cursor, ry1, x1, ry2)
        if _is_nonempty(before) and _has_visible_ink(components, before):
            segments.extend(_text_other_segments(
                before,
                components,
                tokens=tokens,
                component_owners=component_owners,
            ))
        segments.append(RoutingSegment(
            kind="text_latin",
            bbox=bbox,
            text=_latin_group_fallback_text(group_tokens, tokens),
            ppocr_latin_tokens=tuple(
                PpOcrLatinTokenObservation(text=token.text, bbox=token_bbox)
                for token, token_bbox in group_items
            ),
        ))
        cursor = x2
    after = (cursor, ry1, rx2, ry2)
    if _is_nonempty(after) and _has_visible_ink(components, after):
        segments.extend(_text_other_segments(
            after,
            components,
            tokens=tokens,
            component_owners=component_owners,
        ))
    return RoutePartition(tuple(segments))


def _latin_group_fallback_text(
    group_tokens: list[PpOcrV6WordBox],
    all_tokens: tuple[PpOcrV6WordBox, ...],
) -> str:
    """Preserve explicit PP whitespace for an empty-native Latin fallback."""
    ordered = sorted(group_tokens, key=lambda token: token.token_index)
    by_index = {token.token_index: token for token in all_tokens}
    parts: list[str] = []
    for index, token in enumerate(ordered):
        if index:
            previous = ordered[index - 1]
            between = [
                by_index[token_index]
                for token_index in range(previous.token_index + 1, token.token_index)
                if token_index in by_index
            ]
            if any(str(item.text or "").isspace() for item in between):
                parts.append(" ")
        parts.append(str(token.text or ""))
    return "".join(parts)


def _text_other_segments(
    bbox: XYXY,
    components: list[tuple[int, int, int, int, int]],
    *,
    tokens: tuple[PpOcrV6WordBox, ...],
    component_owners: dict[tuple[int, int, int, int, int], int | None],
) -> list[RoutingSegment]:
    """Split a symbol-only gap at measured whitespace between symbol glyphs.

    PP token text contributes only the expected glyph count.  LineCut remains
    responsible for the recognized text and final character geometry.
    """
    gap_components = [
        component
        for component in components
        if _intersect(component[:4], bbox) is not None
    ]
    if not gap_components:
        return []
    token_by_index = {token.token_index: token for token in tokens}
    gap_owners = {
        owner
        for component in gap_components
        for owner in [component_owners.get(component)]
        if owner is not None
    }
    missing_owners = gap_owners.difference(token_by_index)
    if missing_owners:
        raise RuntimeError(
            f"component owner has no PP-OCR token: {sorted(missing_owners)!r}"
        )
    if not gap_owners or any(
        _token_branch(token_by_index[owner].text) != "symbol"
        for owner in gap_owners
    ):
        return [RoutingSegment(kind="text_other", bbox=bbox)]
    if any(component_owners.get(component) is None for component in gap_components):
        return [RoutingSegment(kind="text_other", bbox=bbox)]

    clusters_with_candidates: list[
        tuple[list[tuple[int, int, int, int, int]], str, str]
    ] = []
    expected_glyph_count = 0
    for owner in sorted(
        gap_owners,
        key=lambda index: (_clip(token_by_index[index].bbox, bbox)[0], index),
    ):
        token = token_by_index[owner]
        owned = sorted(
            (component for component in gap_components if component_owners.get(component) == owner),
            key=lambda component: (component[0], component[1]),
        )
        if not owned:
            continue
        glyphs = [char for char in str(token.text or "") if not char.isspace()]
        route_texts = _symbol_route_texts(str(token.text or ""))
        glyph_count = max(1, len(glyphs))
        expected_glyph_count += glyph_count
        split_clusters = _split_components_by_largest_gaps(owned, glyph_count)
        if len(split_clusters) != len(glyphs):
            return [RoutingSegment(kind="text_other", bbox=bbox)]
        clusters_with_candidates.extend(zip(split_clusters, glyphs, route_texts))
    clusters_with_candidates.sort(key=lambda item: (
        min(component[0] for component in item[0]),
        min(component[1] for component in item[0]),
    ))
    clusters = [cluster for cluster, _candidate, _route_text in clusters_with_candidates]
    candidates = [candidate for _cluster, candidate, _route_text in clusters_with_candidates]
    route_texts = [route_text for _cluster, _candidate, route_text in clusters_with_candidates]
    punctuation_candidates = [_punctuation_candidate(candidate) for candidate in candidates]
    if any(not candidate for candidate in punctuation_candidates):
        return [RoutingSegment(kind="text_other", bbox=bbox)]
    if len(clusters) != expected_glyph_count:
        return [RoutingSegment(kind="text_other", bbox=bbox)]
    if len(clusters) == 1:
        content_bbox = _union([component[:4] for component in clusters[0]])
        return [RoutingSegment(
            kind="text_symbol",
            bbox=bbox,
            text=route_texts[0],
            content_bbox=content_bbox,
            component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
            ppocr_punctuation_candidate=punctuation_candidates[0],
        )]

    cluster_boxes = [_union([component[:4] for component in cluster]) for cluster in clusters]
    boundaries: list[int] = []
    for left, right in zip(cluster_boxes, cluster_boxes[1:]):
        if right[0] <= left[2]:
            return [RoutingSegment(kind="text_other", bbox=bbox)]
        boundaries.append((left[2] + right[0]) // 2)
    edges = [bbox[0], *boundaries, bbox[2]]
    return [
        RoutingSegment(
            kind="text_symbol",
            bbox=(edges[index], bbox[1], edges[index + 1], bbox[3]),
            text=route_texts[index],
            content_bbox=cluster_boxes[index],
            component_grouping=COMPONENT_GROUPING_SINGLE_GLYPH,
            ppocr_punctuation_candidate=punctuation_candidates[index],
        )
        for index in range(len(edges) - 1)
        if edges[index + 1] > edges[index]
        and _has_visible_ink(components, (edges[index], bbox[1], edges[index + 1], bbox[3]))
    ]


def _split_components_by_largest_gaps(
    components: list[tuple[int, int, int, int, int]],
    glyph_count: int,
) -> list[list[tuple[int, int, int, int, int]]]:
    if glyph_count <= 1 or len(components) <= 1:
        return [components]
    gaps = [
        (components[index + 1][0] - components[index][2], index)
        for index in range(len(components) - 1)
        if components[index + 1][0] > components[index][2]
    ]
    split_after = {
        index
        for _gap, index in sorted(gaps, key=lambda item: (-item[0], item[1]))[:glyph_count - 1]
    }
    clusters: list[list[tuple[int, int, int, int, int]]] = []
    current: list[tuple[int, int, int, int, int]] = []
    for index, component in enumerate(components):
        current.append(component)
        if index in split_after:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    return clusters


def _punctuation_candidate(value: str) -> str:
    return value if len(value) == 1 and category(value).startswith("P") else ""


def _symbol_route_texts(value: str) -> list[str]:
    """Attach PP-observed whitespace to the preceding punctuation glyph."""
    output: list[str] = []
    leading = ""
    for char in value:
        if char.isspace():
            if output:
                output[-1] += char
            else:
                leading += char
            continue
        output.append(f"{leading}{char}")
        leading = ""
    return output


def _latin_mask_bbox(
    components: list[tuple[int, int, int, int, int]],
    token: PpOcrV6WordBox,
    component_owners: dict[tuple[int, int, int, int, int], int | None],
    region_bbox: XYXY,
) -> XYXY | None:
    seed_bbox = _latin_seed_bbox(token, region_bbox)
    owned = [
        component
        for component in components
        if component_owners.get(component) == token.token_index
        and _intersect(component[:4], seed_bbox) is not None
    ]
    if owned:
        return _union([component[:4] for component in owned])

    # A low-resolution glyph can be physically connected to adjacent
    # punctuation, for example ``(h)``.  Whole-component ownership must remain
    # unique, but the Latin proposal can still recover its own intersecting
    # pixels without taking the punctuation owner's full component.
    core_bbox = _clip(token.bbox, region_bbox)
    fragments = [
        overlap
        for component in components
        if not _is_rule_like_component(component, region_bbox)
        for overlap in [_intersect(component[:4], core_bbox)]
        if overlap is not None
    ]
    return _union(fragments) if fragments else None


def _latin_seed_bbox(token: PpOcrV6WordBox, region_bbox: XYXY) -> XYXY:
    """Expand a Latin proposal by one measured glyph advance in its own row."""
    core = _clip(token.bbox, region_bbox)
    glyph_advance = _latin_glyph_advance(token, region_bbox)
    vertical_margin = max(1, ceil((core[3] - core[1]) / 5))
    return _clip(
        (
            core[0] - glyph_advance,
            core[1] - vertical_margin,
            core[2] + glyph_advance,
            core[3] + vertical_margin,
        ),
        region_bbox,
    )


def _component_owner_token_indices(
    components: list[tuple[int, int, int, int, int]],
    tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
) -> dict[tuple[int, int, int, int, int], int | None]:
    """Assign every ink component to at most one PP-OCR token.

    Word boxes are approximate and commonly meet on the wrong side of a narrow
    glyph.  Exact center containment is authoritative.  A symbol then anchors
    to ink carrying its proposal center, or to the nearest unowned component,
    and reclaims its horizontally connected component stack.  A displaced
    side-by-side component is reclaimed only when it is closer to that symbol
    anchor than to the remaining components of its current Latin owner.  This
    completes multi-part marks without taking the following ``i`` or ``C``.
    Remaining ink goes to the nearest non-symbol token inside that token's
    measured search window.
    """
    owners: dict[tuple[int, int, int, int, int], int | None] = {
        component: None for component in components
    }
    token_branches = {token.token_index: _token_branch(token.text) for token in tokens}
    eligible_components = [
        component
        for component in components
        if not _is_rule_like_component(component, region_bbox)
    ]

    # First preserve the strongest fact: the component center is inside a raw
    # PP-OCR proposal.  The tie-breaker makes shared proposal seams stable.
    for component in eligible_components:
        centered: list[tuple[float, int, float, int]] = []
        center_x = (component[0] + component[2]) / 2.0
        for token in tokens:
            token_bbox = _clip(token.bbox, region_bbox)
            if not _is_nonempty(token_bbox) or not _center_inside(component[:4], token_bbox):
                continue
            token_center_x = (token_bbox[0] + token_bbox[2]) / 2.0
            centered.append((
                -_overlap_ratio(token_bbox, component[:4]),
                token_bbox[2] - token_bbox[0],
                abs(center_x - token_center_x),
                token.token_index,
            ))
        if centered:
            owners[component] = min(centered)[3]

    # Sideways symbol recovery must be evaluated against the immutable PP-OCR
    # ownership facts above.  Reusing later reclaimed owners lets a symbol walk
    # through a neighboring word one component at a time.
    initial_owners = dict(owners)

    for token in tokens:
        if _token_branch(token.text) != "symbol":
            continue
        token_bbox = _clip(token.bbox, region_bbox)
        seed_bbox = _symbol_seed_bbox(token, region_bbox)
        anchors = [
            component
            for component in eligible_components
            if owners[component] == token.token_index
        ]
        candidates = [
            component
            for component in eligible_components
            if owners[component] != token.token_index
            and (
                owners[component] is None
                or token_branches.get(owners[component]) == "latin"
            )
            and _intersect(component[:4], seed_bbox) is not None
        ]
        if not anchors:
            unowned = [
                component
                for component in candidates
                if owners[component] is None
            ]
            if not unowned:
                continue
            token_center_x = (token_bbox[0] + token_bbox[2]) / 2.0
            token_center_y = (token_bbox[1] + token_bbox[3]) / 2.0
            center_carriers = [
                component
                for component in unowned
                if component[0] <= token_center_x <= component[2]
                and component[1] <= token_center_y <= component[3]
            ]
            if center_carriers:
                unowned = center_carriers
            else:
                unowned = [
                    component
                    for component in unowned
                    if _nearest_token_index(component[:4], tokens, region_bbox) == token.token_index
                ]
                if not unowned:
                    continue
            anchor = min(
                unowned,
                key=lambda component: (
                    -_overlap_ratio(token_bbox, component[:4]),
                    _horizontal_gap(component[:4], token_bbox),
                    abs((component[0] + component[2]) - (token_bbox[0] + token_bbox[2])),
                    component[1],
                ),
            )
            owners[anchor] = token.token_index
            anchors = [anchor]

        # Multi-part marks (%, =, :, !) can have one detached component land
        # inside an adjacent word proposal when PP geometry is shifted.  Their
        # components share a horizontal projection, unlike the following C in
        # `,China`, so reclaim that vertical stack as one symbol owner.
        remaining = list(candidates)
        while True:
            symbol_components = [
                component
                for component in remaining
                if any(
                    min(component[2], anchor[2]) > max(component[0], anchor[0])
                    for anchor in anchors
                )
            ]
            if not symbol_components:
                break
            for component in symbol_components:
                owners[component] = token.token_index
                anchors.append(component)
                remaining.remove(component)

        # Compare each side-by-side component once with the original PP token
        # ownership.  This recovers a shifted two-part quote, but a reclaimed
        # component never becomes a stepping stone into an adjacent word.
        reclaimed: list[tuple[int, int, int, int, int]] = []
        stable_symbol_anchors = tuple(anchors)
        for component in remaining:
            owner = initial_owners[component]
            symbol_score = min(
                _component_neighbor_score(component, anchor)
                for anchor in stable_symbol_anchors
            )
            if owner is None:
                latin_anchors = [
                    peer
                    for peer in eligible_components
                    if token_branches.get(initial_owners[peer]) == "latin"
                ]
                if (
                    latin_anchors
                    and symbol_score[0] < 0.0
                    and symbol_score < min(
                        _component_neighbor_score(component, peer)
                        for peer in latin_anchors
                    )
                ):
                    reclaimed.append(component)
                continue
            latin_peers = [
                peer
                for peer in eligible_components
                if peer is not component and initial_owners[peer] == owner
            ]
            if not latin_peers:
                continue
            latin_score = min(
                _component_neighbor_score(component, peer)
                for peer in latin_peers
            )
            if symbol_score < latin_score:
                reclaimed.append(component)
        for component in reclaimed:
            owners[component] = token.token_index

    for component in eligible_components:
        if owners[component] is not None:
            continue
        candidates: list[tuple[int, float, int]] = []
        center_x = (component[0] + component[2]) / 2.0
        for token in tokens:
            if _token_branch(token.text) == "symbol":
                continue
            search_bbox = (
                _latin_seed_bbox(token, region_bbox)
                if _token_branch(token.text) == "latin"
                else _clip(token.bbox, region_bbox)
            )
            if _intersect(component[:4], search_bbox) is None:
                continue
            token_bbox = _clip(token.bbox, region_bbox)
            token_center_x = (token_bbox[0] + token_bbox[2]) / 2.0
            candidates.append((
                _horizontal_gap(component[:4], token_bbox),
                abs(center_x - token_center_x),
                token.token_index,
            ))
        if candidates:
            owners[component] = min(candidates)[2]
    return owners


def _symbol_seed_bbox(token: PpOcrV6WordBox, region_bbox: XYXY) -> XYXY:
    core = _clip(token.bbox, region_bbox)
    width = max(1, core[2] - core[0])
    vertical_margin = max(1, ceil((core[3] - core[1]) / 5))
    return _clip(
        (core[0] - width, core[1] - vertical_margin, core[2] + width, core[3] + vertical_margin),
        region_bbox,
    )


def _nearest_token_index(
    component: XYXY,
    tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
) -> int | None:
    center_x = (component[0] + component[2]) / 2.0
    candidates: list[tuple[float, int]] = []
    for token in tokens:
        token_bbox = _clip(token.bbox, region_bbox)
        if not _is_nonempty(token_bbox):
            continue
        token_center_x = (token_bbox[0] + token_bbox[2]) / 2.0
        candidates.append((abs(center_x - token_center_x), token.token_index))
    return min(candidates)[1] if candidates else None


def _latin_glyph_advance(token: PpOcrV6WordBox, region_bbox: XYXY) -> int:
    core = _clip(token.bbox, region_bbox)
    glyph_count = max(1, sum(1 for char in str(token.text or "") if char.isascii() and char.isalnum()))
    return max(1, ceil((core[2] - core[0]) / glyph_count))


def _has_visible_ink(
    components: list[tuple[int, int, int, int, int]],
    bbox: XYXY,
) -> bool:
    return any(_intersect(component[:4], bbox) is not None for component in components)


def _ink_components(image_bgr: np.ndarray, bbox: XYXY) -> list[tuple[int, int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    page_height, page_width = gray.shape[:2]
    x1, y1, x2, y2 = _clip(bbox, (0, 0, page_width, page_height))
    if x2 <= x1 or y2 <= y1:
        return []
    binary = _foreground_mask(gray[y1:y2, x1:x2])
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(binary, 8)
    return [
        (x1 + left, y1 + top, x1 + left + width, y1 + top + height, area)
        for left, top, width, height, area in (tuple(int(value) for value in stats[index]) for index in range(1, count))
        if area >= _MIN_COMPONENT_AREA
    ]


def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    """Return line foreground for either dark-on-light or light-on-dark text."""
    if crop.size == 0:
        return np.zeros(crop.shape, dtype=np.uint8)
    threshold, _unused = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = float(np.median(border))
    if background <= threshold:
        return (crop > threshold).astype(np.uint8)
    return (crop <= threshold).astype(np.uint8)


def _is_rule_like_component(
    component: tuple[int, int, int, int, int],
    region_bbox: XYXY,
) -> bool:
    """Keep long horizontal rules out of Latin ownership.

    The component remains in the line's visible-ink set, so it can still form a
    LineCut region.  It is only forbidden from widening an EngCut mask across
    neighboring tokens.
    """
    width = component[2] - component[0]
    height = component[3] - component[1]
    region_width = max(1, region_bbox[2] - region_bbox[0])
    region_height = max(1, region_bbox[3] - region_bbox[1])
    return (
        width >= max(8, ceil(region_width * 0.20))
        and width >= max(12, height * 12)
        and height <= max(4, ceil(region_height * 0.12))
    )


def _token_branch(text: str) -> str:
    value = str(text or "")
    if _has_latin_or_digit(value) and not any(is_cjk_char(char) for char in value):
        return "latin"
    if any(is_cjk_char(char) for char in value):
        return "other"
    return "symbol"


def _has_latin_or_digit(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in str(text or ""))


def _nonspace_glyph_count(text: str) -> int:
    return sum(1 for char in str(text or "") if not char.isspace())


def _clip(bbox: XYXY, bounds: XYXY) -> XYXY:
    return (
        max(bbox[0], bounds[0]),
        max(bbox[1], bounds[1]),
        min(bbox[2], bounds[2]),
        min(bbox[3], bounds[3]),
    )


def _intersect(left: XYXY, right: XYXY) -> XYXY | None:
    bbox = _clip(left, right)
    return bbox if _is_nonempty(bbox) else None


def _is_nonempty(bbox: XYXY) -> bool:
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _union(boxes: list[XYXY]) -> XYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _center_inside(component: XYXY, bbox: XYXY) -> bool:
    center_x = (component[0] + component[2]) / 2
    center_y = (component[1] + component[3]) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _overlap_ratio(owner: XYXY, component: XYXY) -> float:
    overlap = _intersect(owner, component)
    if overlap is None:
        return 0.0
    area = max(1, (component[2] - component[0]) * (component[3] - component[1]))
    return ((overlap[2] - overlap[0]) * (overlap[3] - overlap[1])) / area


def _horizontal_gap(left: XYXY, right: XYXY) -> int:
    if left[2] < right[0]:
        return right[0] - left[2]
    if right[2] < left[0]:
        return left[0] - right[2]
    return 0


def _component_neighbor_score(
    left: tuple[int, int, int, int, int],
    right: tuple[int, int, int, int, int],
) -> tuple[float, int, float]:
    left_center = (left[0] + left[2]) / 2.0
    right_center = (right[0] + right[2]) / 2.0
    overlap_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    max_height = max(1, left[3] - left[1], right[3] - right[1])
    return (
        -(overlap_height / max_height),
        _horizontal_gap(left[:4], right[:4]),
        abs(left_center - right_center),
    )


__all__ = ["RoutePartition", "RoutePartitionIssue", "partition_charocr_text_region"]
