"""Build CharOCR text crops from PP-OCRv6 line observations.

PP-OCR word boxes are geometry proposals for Latin/digit routing. For every
line that contains Latin letters or digits, Latin/digit proposals create
EngCut masks and every remaining horizontal region is dispatched to LineCut.
Symbol glyphs may also carry foreground-measured observations; they never
resize or merge native character geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.geometry.foreground import ForegroundComponent, analyze_foreground_components
from app.models.charocr_routing import (
    PpOcrLatinTokenObservation,
    PpOcrSymbolObservation,
    RoutingSegment,
)
from app.core.text_classification import is_cjk_char


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class RoutePartitionIssue:
    code: str
    message: str
    bbox: XYXY


@dataclass(frozen=True)
class RoutePartitionDiagnostic:
    code: str
    message: str
    bbox: XYXY


@dataclass(frozen=True)
class RoutePartition:
    segments: tuple[RoutingSegment, ...]
    issues: tuple[RoutePartitionIssue, ...] = ()
    symbol_observations: tuple[PpOcrSymbolObservation, ...] = ()
    diagnostics: tuple[RoutePartitionDiagnostic, ...] = ()


def partition_charocr_text_region(
    image_bgr: np.ndarray | None,
    prepass_line: PpOcrV6LineHint,
    region_bbox: XYXY,
    *,
    excluded_bboxes: tuple[XYXY, ...] = (),
    linecut_owned_bboxes: tuple[XYXY, ...] = (),
) -> RoutePartition:
    """Partition one non-structural PP-OCR row region into native OCR crops.

    CJK-only rows remain one LineCut route. Rows containing Latin or digits use
    PP-OCR word boxes as EngCut mask proposals, including otherwise pure Latin
    rows. Component closure may repair a word-box edge, but ink anchored by
    another PP-OCR token center cannot enter a Latin mask. Everything outside
    those masks remains a LineCut region.
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

    components = list(
        analyze_foreground_components(image_bgr, region_bbox).components
    )
    if not components:
        return RoutePartition(())

    crossing = _component_crossing_owned_boundary(components, linecut_owned_bboxes)
    if crossing is not None:
        return RoutePartition((), (
            RoutePartitionIssue(
                code="linecut_ownership_boundary_crosses_glyph",
                message="explicit LineCut ownership boundary crosses one foreground component",
                bbox=crossing,
            ),
        ))

    tokens = tuple(
        token
        for token in prepass_line.words
        if str(token.text or "").strip() and _intersect(token.bbox, region_bbox) is not None
        and not _token_has_only_excluded_ink(
            token,
            components,
            region_bbox,
            excluded_bboxes,
        )
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

    latin_tokens = tuple(
        token
        for token in tokens
        if _token_route_branch(token, linecut_owned_bboxes) == "latin"
    )
    if not latin_tokens:
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox),))

    (
        component_owners,
        ownership_issues,
        ownership_diagnostics,
    ) = _component_owner_token_indices(
        components,
        tokens,
        region_bbox,
        linecut_owned_bboxes=linecut_owned_bboxes,
    )
    if ownership_issues:
        return RoutePartition(
            (),
            ownership_issues,
            diagnostics=ownership_diagnostics,
        )
    masks: list[tuple[PpOcrV6WordBox, XYXY]] = []
    diagnostics = list(ownership_diagnostics)
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
            diagnostics.append(RoutePartitionDiagnostic(
                code="missing_latin_token_ink",
                message=(
                    "PP-OCRv6 Latin/digit token has no unambiguous ink; "
                    f"remaining foreground stays on the LineCut route: {token.text!r}"
                ),
                bbox=_clip(token.bbox, region_bbox),
            ))
            continue
        token_bbox = _clip(token.bbox, region_bbox)
        if _intersect(mask_bbox, token_bbox) is None:
            diagnostics.append(RoutePartitionDiagnostic(
                code="disjoint_latin_token_ink",
                message=(
                    "PP-OCRv6 Latin/digit token owns no ink inside its observed "
                    f"bbox; remaining foreground stays on the LineCut route: {token.text!r}"
                ),
                bbox=token_bbox,
            ))
            continue
        masks.append((token, mask_bbox))

    routed = _segments_from_latin_masks(
        region_bbox,
        masks,
        components,
    )
    return RoutePartition(
        segments=routed.segments,
        issues=routed.issues,
        symbol_observations=_single_symbol_observations(
            components,
            tokens,
            component_owners,
            region_bbox,
        ),
        diagnostics=tuple(diagnostics),
    )


def _single_symbol_observations(
    components: list[ForegroundComponent],
    tokens: tuple[PpOcrV6WordBox, ...],
    component_owners: dict[ForegroundComponent, int | None],
    region_bbox: XYXY,
) -> tuple[PpOcrSymbolObservation, ...]:
    """Measure symbol glyphs from the same ownership used by routing.

    PP-OCR may place several punctuation glyphs in one token.  The token text
    supplies their count and order; its proposal bbox supplies their ordered
    horizontal cells.  Every cell must own foreground before any observation
    from that token is emitted, so incomplete topology is never guessed.
    """
    observations: list[PpOcrSymbolObservation] = []
    for token in sorted(tokens, key=lambda item: item.token_index):
        text = str(token.text or "")
        glyphs = tuple(char for char in text if not char.isspace())
        if not glyphs or any(char.isalnum() for char in glyphs):
            continue
        owned = [
            component
            for component in components
            if component_owners.get(component) == token.token_index
        ]
        proposal_bbox = _clip(token.bbox, region_bbox)
        if not owned or not any(
            _intersect(component.bbox, proposal_bbox) is not None
            for component in owned
        ):
            continue
        if proposal_bbox[2] - proposal_bbox[0] < len(glyphs):
            continue
        glyph_cells = _ordered_glyph_cells(proposal_bbox, len(glyphs))
        glyph_components: list[list[XYXY]] = [[] for _glyph in glyphs]
        for component in owned:
            center_x = (component.bbox[0] + component.bbox[2]) / 2.0
            cell_index = next((
                index
                for index, cell in enumerate(glyph_cells)
                if cell[0] <= center_x < cell[2]
            ), 0 if center_x < glyph_cells[0][0] else len(glyph_cells) - 1)
            glyph_components[cell_index].append(component.bbox)
        if any(not boxes for boxes in glyph_components):
            continue
        observations.extend(
            PpOcrSymbolObservation(
                text=glyph,
                bbox=_union(boxes),
                proposal_bbox=glyph_cells[index],
                leading_space=index == 0 and text[:1].isspace(),
                trailing_space=index == len(glyphs) - 1 and text[-1:].isspace(),
            )
            for index, (glyph, boxes) in enumerate(zip(glyphs, glyph_components))
        )
    return tuple(observations)


def _ordered_glyph_cells(proposal_bbox: XYXY, glyph_count: int) -> tuple[XYXY, ...]:
    """Project one ordered PP token bbox into non-overlapping glyph cells."""
    x1, y1, x2, y2 = proposal_bbox
    return tuple(
        (
            round(x1 + (x2 - x1) * index / glyph_count),
            y1,
            round(x1 + (x2 - x1) * (index + 1) / glyph_count),
            y2,
        )
        for index in range(glyph_count)
    )


def _token_has_only_excluded_ink(
    token: PpOcrV6WordBox,
    components: list[ForegroundComponent],
    region_bbox: XYXY,
    excluded_bboxes: tuple[XYXY, ...],
) -> bool:
    token_bbox = _clip(token.bbox, region_bbox)
    if not any(_intersect(token_bbox, bbox) is not None for bbox in excluded_bboxes):
        return False
    return not any(
        _intersect(component.bbox, token_bbox) is not None
        for component in components
    )


def _segments_from_latin_masks(
    region_bbox: XYXY,
    masks: list[tuple[PpOcrV6WordBox, XYXY]],
    components: list[ForegroundComponent],
) -> RoutePartition:
    ordered_masks = sorted(masks, key=lambda item: (item[1][0], item[0].token_index))
    rx1, ry1, rx2, ry2 = region_bbox
    for (_left_token, left_bbox), (_right_token, right_bbox) in zip(
        ordered_masks,
        ordered_masks[1:],
    ):
        if right_bbox[0] < left_bbox[2]:
            return RoutePartition((), (
                RoutePartitionIssue(
                    code="overlapping_latin_masks",
                    message="PP-OCRv6 Latin/digit masks overlap after component closure",
                    bbox=(right_bbox[0], ry1, left_bbox[2], ry2),
                ),
            ))

    segments: list[RoutingSegment] = []
    cursor = rx1
    for token, bbox in ordered_masks:
        x1, _y1, x2, _y2 = bbox
        before = (cursor, ry1, x1, ry2)
        if _is_nonempty(before) and _has_visible_ink(components, before):
            segments.append(RoutingSegment(kind="text_other", bbox=before))
        segments.append(RoutingSegment(
            kind="text_latin",
            bbox=bbox,
            text=str(token.text or ""),
            ppocr_latin_tokens=(PpOcrLatinTokenObservation(
                text=token.text,
                bbox=_clip(token.bbox, region_bbox),
            ),),
        ))
        cursor = x2
    after = (cursor, ry1, rx2, ry2)
    if _is_nonempty(after) and _has_visible_ink(components, after):
        segments.append(RoutingSegment(kind="text_other", bbox=after))
    return RoutePartition(tuple(segments))


def _latin_mask_bbox(
    components: list[ForegroundComponent],
    token: PpOcrV6WordBox,
    component_owners: dict[ForegroundComponent, int | None],
    region_bbox: XYXY,
) -> XYXY | None:
    owned_boxes = [
        component.bbox
        for component in components
        if component_owners.get(component) == token.token_index
    ]
    return _union(owned_boxes) if owned_boxes else None


def _component_owner_token_indices(
    components: list[ForegroundComponent],
    tokens: tuple[PpOcrV6WordBox, ...],
    region_bbox: XYXY,
    *,
    linecut_owned_bboxes: tuple[XYXY, ...],
) -> tuple[
    dict[ForegroundComponent, int | None],
    tuple[RoutePartitionIssue, ...],
    tuple[RoutePartitionDiagnostic, ...],
]:
    """Assign components once by ordered PP token ownership cells.

    Cell boundaries are observations derived from adjacent PP boxes. A
    component crossing multiple token centers remains ambiguous unless the
    surrounding ownership produces one forced token/component match.
    """
    ordered = tuple(
        token
        for token in sorted(tokens, key=lambda item: (item.bbox[0], item.token_index))
        if str(token.text or "").strip()
    )
    if not ordered:
        return {component: None for component in components}, (), ()
    boundaries = [region_bbox[0]]
    for left, right in zip(ordered, ordered[1:]):
        boundaries.append((left.bbox[2] + right.bbox[0]) / 2.0)
    boundaries.append(region_bbox[2])
    token_centers = {
        token.token_index: (token.bbox[0] + token.bbox[2]) / 2.0
        for token in ordered
    }

    owners: dict[ForegroundComponent, int | None] = {}
    ambiguous_components: set[ForegroundComponent] = set()
    crossed_tokens_by_component: dict[ForegroundComponent, tuple[int, ...]] = {}
    center_owned_components: set[ForegroundComponent] = set()
    linecut_owned_components: set[ForegroundComponent] = set()
    for component in components:
        if _component_center_in_any_bbox(component, linecut_owned_bboxes):
            owners[component] = None
            linecut_owned_components.add(component)
            continue
        crossed_centers = [
            token_index
            for token_index, center_x in token_centers.items()
            if component.bbox[0] <= center_x < component.bbox[2]
        ]
        if len(crossed_centers) > 1:
            owners[component] = None
            ambiguous_components.add(component)
            crossed_tokens_by_component[component] = tuple(crossed_centers)
            continue
        if len(crossed_centers) == 1:
            owners[component] = crossed_centers[0]
            center_owned_components.add(component)
            continue
        center_x = (component.bbox[0] + component.bbox[2]) / 2.0
        owner = next((
            token.token_index
            for position, token in enumerate(ordered)
            if boundaries[position] <= center_x < boundaries[position + 1]
        ), None)
        owners[component] = owner

    token_by_index = {token.token_index: token for token in ordered}

    # Resolve only forced token/component matches. If an ambiguous component
    # crosses several token centers, it belongs to the sole token that still
    # has no other component, provided no second ambiguous component competes
    # for that token. This preserves an unresolved fused glyph when both sides
    # are missing instead of choosing by distance.
    while ambiguous_components:
        owned_token_indices = {
            owner for owner in owners.values() if owner is not None
        }
        forced_by_token: dict[int, list[ForegroundComponent]] = {}
        for component in ambiguous_components:
            crossed = crossed_tokens_by_component[component]
            missing = [index for index in crossed if index not in owned_token_indices]
            if len(missing) == 1 and len(missing) < len(crossed):
                forced_by_token.setdefault(missing[0], []).append(component)
        resolved = {
            candidates[0]: token_index
            for token_index, candidates in forced_by_token.items()
            if len(candidates) == 1
        }
        if not resolved:
            break
        for component, token_index in resolved.items():
            owners[component] = token_index
            ambiguous_components.remove(component)

    # A detached symbol part may be recovered only when it intersects the PP
    # proposal itself and shares horizontal projection with an already owned
    # part. There is no recursive walk beyond the observed proposal.
    for token in ordered:
        if _token_route_branch(token, linecut_owned_bboxes) != "symbol":
            continue
        proposal_bbox = _clip(token.bbox, region_bbox)
        anchors = [
            component
            for component in components
            if owners[component] == token.token_index
        ]
        for component in components:
            if owners[component] == token.token_index:
                continue
            if component in linecut_owned_components:
                continue
            if component in center_owned_components:
                continue
            if _intersect(component.bbox, proposal_bbox) is None:
                continue
            if any(
                max(component.bbox[0], anchor.bbox[0])
                < min(component.bbox[2], anchor.bbox[2])
                for anchor in anchors
            ):
                owners[component] = token.token_index

    _reclaim_latin_glyph_parts(
        components,
        ordered,
        owners,
        linecut_owned_components=linecut_owned_components,
    )
    completion_issues, completion_diagnostics = _complete_single_latin_token_components(
        components,
        ordered,
        owners,
        region_bbox,
        center_owned_components=center_owned_components,
        linecut_owned_components=linecut_owned_components,
        ambiguous_components=ambiguous_components,
    )

    # If a non-symbol token is empty, a unique component intersecting its raw
    # PP observation is the only additional ownership fact available. Multiple
    # candidates remain unresolved instead of being ranked by proximity.
    for token in ordered:
        if _token_route_branch(token, linecut_owned_bboxes) == "symbol":
            continue
        if any(owner == token.token_index for owner in owners.values()):
            continue
        token_bbox = _clip(token.bbox, region_bbox)
        candidates = [
            component
            for component in components
            if _intersect(component.bbox, token_bbox) is not None
            and component not in ambiguous_components
            and component not in linecut_owned_components
            and (
                owners[component] is None
                or _token_route_branch(
                    token_by_index[owners[component]],
                    linecut_owned_bboxes,
                ) != "symbol"
            )
        ]
        if len(candidates) == 1:
            owners[candidates[0]] = token.token_index
    return owners, completion_issues, completion_diagnostics


def _reclaim_latin_glyph_parts(
    components: list[ForegroundComponent],
    tokens: tuple[PpOcrV6WordBox, ...],
    owners: dict[ForegroundComponent, int | None],
    *,
    linecut_owned_components: set[ForegroundComponent],
) -> None:
    """Keep one Latin glyph intact across a neighboring symbol cell.

    PP symbol boxes often include trailing whitespace. Their midpoint boundary
    can therefore capture a detached stem or descender from the next Latin
    token. A complete foreground component is reassigned only when it shares
    horizontal ink projection with an already-owned Latin component. CJK and
    explicit LineCut ownership are never eligible, and competing Latin claims
    remain unresolved.
    """
    token_by_index = {token.token_index: token for token in tokens}
    latin_indices = {
        token.token_index
        for token in tokens
        if _token_branch(token.text) == "latin"
    }
    anchors = {
        token_index: tuple(
            component
            for component in components
            if owners.get(component) == token_index
        )
        for token_index in latin_indices
    }
    claims: dict[ForegroundComponent, set[int]] = {}
    for component in components:
        if component in linecut_owned_components:
            continue
        current_owner = owners.get(component)
        if current_owner in latin_indices:
            continue
        if current_owner is not None:
            owner_token = token_by_index.get(current_owner)
            if owner_token is None or _token_branch(owner_token.text) != "symbol":
                continue
        for token_index, token_anchors in anchors.items():
            if any(_shares_horizontal_ink(component, anchor) for anchor in token_anchors):
                claims.setdefault(component, set()).add(token_index)
    for component, token_indices in claims.items():
        if len(token_indices) == 1:
            owners[component] = next(iter(token_indices))


def _shares_horizontal_ink(
    left: ForegroundComponent,
    right: ForegroundComponent,
) -> bool:
    return max(left.bbox[0], right.bbox[0]) < min(left.bbox[2], right.bbox[2])


def _complete_single_latin_token_components(
    components: list[ForegroundComponent],
    tokens: tuple[PpOcrV6WordBox, ...],
    owners: dict[ForegroundComponent, int | None],
    region_bbox: XYXY,
    *,
    center_owned_components: set[ForegroundComponent],
    linecut_owned_components: set[ForegroundComponent],
    ambiguous_components: set[ForegroundComponent],
) -> tuple[
    tuple[RoutePartitionIssue, ...],
    tuple[RoutePartitionDiagnostic, ...],
]:
    """Complete one PP single-character token before its EngCut mask is built.

    A tiny owned component can be the detached dot of an italic glyph while
    the stem was assigned only by a midpoint cell. A unique weak component
    intersecting the same PP proposal may complete that token. Strong token
    centers, explicit LineCut ownership, symbols, and competing Latin claims
    remain untouched.
    """

    token_by_index = {token.token_index: token for token in tokens}
    latin_tokens = tuple(
        token
        for token in tokens
        if _token_branch(token.text) == "latin"
    )
    issues: list[RoutePartitionIssue] = []
    diagnostics: list[RoutePartitionDiagnostic] = []
    for token in latin_tokens:
        if len(str(token.text or "").strip()) != 1:
            continue
        anchors = [
            component
            for component in components
            if owners.get(component) == token.token_index
        ]
        if not anchors:
            continue
        proposal_bbox = _clip(token.bbox, region_bbox)
        proposal_height = proposal_bbox[3] - proposal_bbox[1]
        anchor_bbox = _union([component.bbox for component in anchors])
        anchor_height = anchor_bbox[3] - anchor_bbox[1]
        if proposal_height <= 0 or anchor_height * 3 >= proposal_height:
            continue

        candidates: list[ForegroundComponent] = []
        for component in components:
            if component in anchors:
                continue
            if component in center_owned_components:
                continue
            if component in linecut_owned_components:
                continue
            if component in ambiguous_components:
                continue
            if _intersect(component.bbox, proposal_bbox) is None:
                continue
            current_owner = owners.get(component)
            if current_owner is not None:
                owner_token = token_by_index.get(current_owner)
                if owner_token is None:
                    continue
                if _token_branch(owner_token.text) in {"latin", "symbol"}:
                    continue
            latin_claims = [
                candidate.token_index
                for candidate in latin_tokens
                if _intersect(component.bbox, _clip(candidate.bbox, region_bbox)) is not None
            ]
            if latin_claims != [token.token_index]:
                continue
            completed_bbox = _union([anchor_bbox, component.bbox])
            if completed_bbox[3] - completed_bbox[1] < proposal_height / 2.0:
                continue
            if not _single_token_components_form_one_glyph(
                anchor_bbox,
                component.bbox,
                proposal_bbox,
            ):
                continue
            candidates.append(component)

        if len(candidates) == 1:
            completed_bbox = _union([anchor_bbox, candidates[0].bbox])
            owners[candidates[0]] = token.token_index
            diagnostics.append(RoutePartitionDiagnostic(
                code="single_latin_token_geometry_completed",
                message=(
                    "PP-OCRv6 single-character token recovered one uniquely "
                    f"bound foreground component: {token.text!r}"
                ),
                bbox=completed_bbox,
            ))
            continue
        issues.append(RoutePartitionIssue(
            code="incomplete_single_latin_token_ink",
            message=(
                "PP-OCRv6 single-character token owns only a small foreground "
                "fragment and cannot be completed uniquely: "
                f"{token.text!r} candidates={len(candidates)}"
            ),
            bbox=proposal_bbox,
        ))
    return tuple(issues), tuple(diagnostics)


def _single_token_components_form_one_glyph(
    anchor_bbox: XYXY,
    candidate_bbox: XYXY,
    proposal_bbox: XYXY,
) -> bool:
    proposal_width = proposal_bbox[2] - proposal_bbox[0]
    proposal_height = proposal_bbox[3] - proposal_bbox[1]
    anchor_center_x = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
    candidate_center_x = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
    candidate_center_y = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
    horizontal_gap = max(
        0,
        max(anchor_bbox[0], candidate_bbox[0])
        - min(anchor_bbox[2], candidate_bbox[2]),
    )
    vertical_gap = max(
        0,
        max(anchor_bbox[1], candidate_bbox[1])
        - min(anchor_bbox[3], candidate_bbox[3]),
    )
    completed_bbox = _union([anchor_bbox, candidate_bbox])
    completed_width = completed_bbox[2] - completed_bbox[0]
    completed_height = completed_bbox[3] - completed_bbox[1]
    return (
        candidate_center_y > anchor_center_y
        and candidate_bbox[3] - candidate_bbox[1]
        > (anchor_bbox[3] - anchor_bbox[1]) * 2
        and horizontal_gap <= max(2, proposal_height // 8)
        and vertical_gap <= max(2, proposal_height // 3)
        and abs(candidate_center_x - anchor_center_x)
        <= max(proposal_width, proposal_height / 2.0)
        and completed_width <= proposal_width + proposal_height / 3.0
        and completed_height <= proposal_height + proposal_height / 3.0
    )


def _has_visible_ink(
    components: list[ForegroundComponent],
    bbox: XYXY,
) -> bool:
    return any(_intersect(component.bbox, bbox) is not None for component in components)


def _token_branch(text: str) -> str:
    value = str(text or "")
    if _has_latin_or_digit(value) and not any(is_cjk_char(char) for char in value):
        return "latin"
    if any(is_cjk_char(char) for char in value):
        return "other"
    return "symbol"


def _token_route_branch(
    token: PpOcrV6WordBox,
    linecut_owned_bboxes: tuple[XYXY, ...],
) -> str:
    if _token_center_in_any_bbox(token, linecut_owned_bboxes):
        return "other"
    return _token_branch(token.text)


def _token_center_in_any_bbox(
    token: PpOcrV6WordBox,
    bboxes: tuple[XYXY, ...],
) -> bool:
    center_x = (token.bbox[0] + token.bbox[2]) / 2.0
    center_y = (token.bbox[1] + token.bbox[3]) / 2.0
    return any(
        bbox[0] <= center_x < bbox[2] and bbox[1] <= center_y < bbox[3]
        for bbox in bboxes
    )


def _component_center_in_any_bbox(
    component: ForegroundComponent,
    bboxes: tuple[XYXY, ...],
) -> bool:
    center_x = (component.bbox[0] + component.bbox[2]) / 2.0
    center_y = (component.bbox[1] + component.bbox[3]) / 2.0
    return any(
        bbox[0] <= center_x < bbox[2] and bbox[1] <= center_y < bbox[3]
        for bbox in bboxes
    )


def _component_crossing_owned_boundary(
    components: list[ForegroundComponent],
    bboxes: tuple[XYXY, ...],
) -> XYXY | None:
    for bbox in bboxes:
        boundary_x = bbox[2]
        for component in components:
            if (
                bbox[1] < component.bbox[3]
                and component.bbox[1] < bbox[3]
                and component.bbox[0] < boundary_x < component.bbox[2]
            ):
                return component.bbox
    return None


def _has_latin_or_digit(text: str) -> bool:
    return any(
        char.isascii() and (char.isalpha() or char.isdigit())
        for char in str(text or "")
    )


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


__all__ = [
    "RoutePartition",
    "RoutePartitionDiagnostic",
    "RoutePartitionIssue",
    "partition_charocr_text_region",
]
