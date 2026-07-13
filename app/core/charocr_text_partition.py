"""Build CharOCR text crops from PP-OCRv6 line observations.

PP-OCR word boxes are geometry proposals for Latin/digit routing. For every
line that contains Latin letters or digits, Latin/digit proposals create
EngCut masks and every remaining horizontal region is dispatched to LineCut.
Punctuation proposals are ownership boundaries, never OCR observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import cv2
import numpy as np

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint, PpOcrV6WordBox
from app.models.charocr_routing import PpOcrLatinTokenObservation, RoutingSegment
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

    CJK-only rows remain one LineCut route. Pure Latin/digit rows keep their
    complete physical row and go directly to EngCut. Mixed rows use PP-OCR
    Latin word boxes as EngCut mask proposals. Component closure may repair a
    word-box edge, but ink anchored by any other PP-OCR token cannot enter a
    Latin mask. Everything outside those masks remains a LineCut region.
    """
    text = str(prepass_line.text or "")
    if not _has_latin_or_digit(text):
        return RoutePartition((RoutingSegment(kind="text_other", bbox=region_bbox, text=text),))
    if not any(is_cjk_char(char) for char in text):
        return RoutePartition((RoutingSegment(
            kind="text_latin",
            bbox=region_bbox,
            text=text,
            ppocr_latin_tokens=tuple(
                PpOcrLatinTokenObservation(
                    text=token.text,
                    bbox=_clip(token.bbox, region_bbox),
                )
                for token in sorted(prepass_line.words, key=lambda item: item.token_index)
                if _token_branch(token.text) == "latin"
                and _is_nonempty(_clip(token.bbox, region_bbox))
            ),
        ),))
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
        if _has_visible_ink(components, gap) or _has_explicit_boundary_token(
            current_tokens[-1][0], token, tokens
        ):
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
            segments.append(RoutingSegment(kind="text_other", bbox=before))
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
        segments.append(RoutingSegment(kind="text_other", bbox=after))
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


def _has_explicit_boundary_token(
    left: PpOcrV6WordBox,
    right: PpOcrV6WordBox,
    tokens: tuple[PpOcrV6WordBox, ...],
) -> bool:
    """Keep EngCut groups separated across an observed non-space boundary."""
    return any(
        left.token_index < token.token_index < right.token_index
        and bool(str(token.text or "").strip())
        and _token_branch(token.text) != "latin"
        for token in tokens
    )


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
    glyph. A component can be owned by CJK or Latin, while punctuation is an
    exclusion boundary. Remaining ink is offered to the nearest PP token; it
    enters EngCut only when that token is Latin/digit.
    """
    owners: dict[tuple[int, int, int, int, int], int | None] = {
        component: None for component in components
    }
    eligible_components = [
        component
        for component in components
        if not _is_rule_like_component(component, region_bbox)
    ]

    # Symbols and whitespace do not own native OCR ink. They remain in the
    # LineCut region; only Latin/digit and CJK token observations participate
    # in component ownership.
    for component in eligible_components:
        centered: list[tuple[float, int, float, int]] = []
        center_x = (component[0] + component[2]) / 2.0
        for token in tokens:
            if _token_branch(token.text) == "symbol":
                continue
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

    # PP word boxes are approximate, but measured overlap is stronger evidence
    # than a neighboring token merely being closer. Punctuation participates
    # in the comparison only as a blocker, so it can never become EngCut text.
    token_by_index = {token.token_index: token for token in tokens}
    for component in eligible_components:
        overlapping = [
            token
            for token in tokens
            if _intersect(component[:4], _clip(token.bbox, region_bbox)) is not None
        ]
        if not overlapping:
            continue
        best = min(overlapping, key=lambda token: (
            -_overlap_ratio(_clip(token.bbox, region_bbox), component[:4]),
            0 if _center_inside(component[:4], _clip(token.bbox, region_bbox)) else 1,
            0 if _token_branch(token.text) == "symbol" else 1,
            token.token_index,
        ))
        best_branch = _token_branch(best.text)
        best_overlap = _overlap_ratio(_clip(best.bbox, region_bbox), component[:4])
        current = owners[component]
        current_overlap = (
            _overlap_ratio(_clip(token_by_index[current].bbox, region_bbox), component[:4])
            if current is not None else 0.0
        )
        if best_overlap > current_overlap or best_branch == "symbol":
            owners[component] = best.token_index

    # Multi-component symbols have vertically separated ink (for example the
    # dots and slash of `%`). They may reclaim only components sharing a
    # horizontal projection with an already owned symbol component. Horizontal
    # walking is forbidden because it previously stole adjacent K/L/h glyphs.
    for token in tokens:
        if _token_branch(token.text) != "symbol":
            continue
        anchors = [
            component for component in eligible_components
            if owners[component] == token.token_index
        ]
        if not anchors:
            continue
        while True:
            reclaimed = [
                component
                for component in eligible_components
                if owners[component] != token.token_index
                and any(
                    min(component[2], anchor[2]) > max(component[0], anchor[0])
                    for anchor in anchors
                )
            ]
            if not reclaimed:
                break
            for component in reclaimed:
                owners[component] = token.token_index
                anchors.append(component)

    for component in eligible_components:
        if owners[component] is not None:
            continue
        candidates: list[tuple[int, float, int, int, str]] = []
        center_x = (component[0] + component[2]) / 2.0
        for token in tokens:
            branch = _token_branch(token.text)
            token_bbox = _clip(token.bbox, region_bbox)
            token_center_x = (token_bbox[0] + token_bbox[2]) / 2.0
            candidates.append((
                _horizontal_gap(component[:4], token_bbox),
                abs(center_x - token_center_x),
                0 if branch == "symbol" else 1,
                token.token_index,
                branch,
            ))
        if candidates:
            _gap, _center_delta, _branch_priority, token_index, branch = min(candidates)
            token = token_by_index[token_index]
            ownership_window = (
                _latin_seed_bbox(token, region_bbox)
                if branch == "latin"
                else _clip(token.bbox, region_bbox)
            )
            if branch != "symbol" and _intersect(component[:4], ownership_window) is not None:
                owners[component] = token_index
    return owners


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


__all__ = ["RoutePartition", "RoutePartitionIssue", "partition_charocr_text_region"]
