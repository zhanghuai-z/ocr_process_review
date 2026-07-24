"""Typed layout routing contract shared by routing producers and consumers."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from app.models.ocr_routing_observation import BlockObservationAlignment


XYXY = tuple[int, int, int, int]
ROUTING_SOURCE_PPOCR_V6_PREPASS = "ppocrv6_prepass"
ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT = "layout_vertical_text"
TEXT_AXIS_HORIZONTAL = "horizontal"
TEXT_AXIS_VERTICAL = "vertical"
ROUTE_SEGMENT_TEXT_OTHER = "text_other"
ROUTE_SEGMENT_TEXT_LATIN = "text_latin"
ROUTE_SEGMENT_FORMULA = "formula"
ROUTE_SEGMENT_DECORATION = "decoration"
ROUTE_SEGMENT_SKIP = "skip"
TEXT_ROUTE_SEGMENT_KINDS = frozenset({
    ROUTE_SEGMENT_TEXT_OTHER,
    ROUTE_SEGMENT_TEXT_LATIN,
})
VALID_ROUTE_SEGMENT_KINDS = frozenset({
    *TEXT_ROUTE_SEGMENT_KINDS,
    ROUTE_SEGMENT_FORMULA,
    ROUTE_SEGMENT_DECORATION,
    ROUTE_SEGMENT_SKIP,
})


@dataclass(frozen=True)
class PpOcrLatinTokenObservation:
    """One PP-OCR Latin/ASCII-digit token retained as a route observation.

    The bbox is PP-OCR's observed token geometry. It may extend beyond the
    foreground-safe EngCut segment mask, but must intersect that segment.
    """

    text: str
    bbox: XYXY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        if not self.text or not any(
            char.isascii() and (char.isalpha() or char.isdigit())
            for char in self.text
        ):
            raise ValueError("PP-OCR Latin token observation requires Latin/ASCII-digit text")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("PP-OCR Latin token observation requires non-empty geometry")


@dataclass(frozen=True)
class PpOcrSymbolObservation:
    """One PP-observed symbol glyph with foreground-measured geometry."""

    text: str
    bbox: XYXY
    proposal_bbox: XYXY
    leading_space: bool = False
    trailing_space: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        object.__setattr__(self, "proposal_bbox", xyxy(self.proposal_bbox))
        if len(self.text) != 1 or self.text.isspace() or self.text.isalnum():
            raise ValueError("PP-OCR symbol observation requires one non-alphanumeric glyph")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("PP-OCR symbol observation requires non-empty foreground geometry")
        if self.proposal_bbox[2] <= self.proposal_bbox[0] or self.proposal_bbox[3] <= self.proposal_bbox[1]:
            raise ValueError("PP-OCR symbol proposal requires non-empty geometry")


@dataclass(frozen=True)
class VlSemanticMarkerObservation:
    """One explicit VL marker bound to foreground-measured page geometry."""

    text: str
    bbox: XYXY
    proposal_bbox: XYXY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        object.__setattr__(self, "proposal_bbox", xyxy(self.proposal_bbox))
        if not self.text or any(char.isspace() for char in self.text):
            raise ValueError("VL semantic marker observation requires non-space text")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("VL semantic marker observation requires non-empty foreground geometry")
        if self.proposal_bbox[2] <= self.proposal_bbox[0] or self.proposal_bbox[3] <= self.proposal_bbox[1]:
            raise ValueError("VL semantic marker proposal requires non-empty geometry")


@dataclass(frozen=True)
class RoutingSegment:
    """One typed two-dimensional region on a physical routing line.

    Regions are not an ordered one-dimensional partition. A structural mask
    may overlap a text candidate rectangle; native-input materialization gives
    the structural region explicit masking precedence. Only a formula may set
    ``content_bbox``. It is page-bounded canonical formula geometry, must
    contain the segment mask, and may extend beyond this line's exact
    intersection in ``bbox``.
    """

    kind: str
    bbox: XYXY
    label: str = ""
    text: str = ""
    structural_block_uid: str = ""
    text_source: str = ""
    content_bbox: XYXY | None = None
    ppocr_latin_tokens: tuple[PpOcrLatinTokenObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalize_route_segment_kind(self.kind))
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        if self.content_bbox is not None:
            object.__setattr__(self, "content_bbox", xyxy(self.content_bbox))
        latin_tokens = tuple(self.ppocr_latin_tokens or ())
        if latin_tokens and self.kind != ROUTE_SEGMENT_TEXT_LATIN:
            raise ValueError("PP-OCR Latin token observations require a text_latin route")
        if any(not isinstance(token, PpOcrLatinTokenObservation) for token in latin_tokens):
            raise TypeError("PP-OCR Latin token observations require typed values")
        object.__setattr__(self, "ppocr_latin_tokens", latin_tokens)


@dataclass(frozen=True)
class RoutingLine:
    index: int
    bbox: XYXY
    segments: tuple[RoutingSegment, ...]
    source: str = ""
    text_axis: str = TEXT_AXIS_HORIZONTAL
    orientation_angle: int = -1
    ppocr_symbol_observations: tuple[PpOcrSymbolObservation, ...] = ()
    vl_marker_observations: tuple[VlSemanticMarkerObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "ppocr_symbol_observations", tuple(self.ppocr_symbol_observations))
        object.__setattr__(self, "vl_marker_observations", tuple(self.vl_marker_observations))
        if any(not isinstance(segment, RoutingSegment) for segment in self.segments):
            raise TypeError("routing line segments require typed values")
        if self.text_axis not in {TEXT_AXIS_HORIZONTAL, TEXT_AXIS_VERTICAL}:
            raise ValueError(f"unsupported routing text axis: {self.text_axis!r}")
        if self.orientation_angle not in {-1, 0, 180}:
            raise ValueError(
                f"unsupported routing textline orientation angle: {self.orientation_angle}"
            )
        if any(
            not isinstance(observation, PpOcrSymbolObservation)
            for observation in self.ppocr_symbol_observations
        ):
            raise TypeError("routing line symbol observations require typed values")
        if any(
            not isinstance(observation, VlSemanticMarkerObservation)
            for observation in self.vl_marker_observations
        ):
            raise TypeError("routing line VL marker observations require typed values")

    @property
    def has_formula(self) -> bool:
        return any(segment.kind == ROUTE_SEGMENT_FORMULA for segment in self.segments)


def is_text_route_segment_kind(kind: str) -> bool:
    return kind in TEXT_ROUTE_SEGMENT_KINDS


def normalize_route_segment_kind(value: object) -> str:
    kind = str(value or ROUTE_SEGMENT_TEXT_OTHER).strip() or ROUTE_SEGMENT_TEXT_OTHER
    if kind not in VALID_ROUTE_SEGMENT_KINDS:
        raise ValueError(f"unsupported layout route segment kind: {kind!r}")
    return kind


@dataclass(frozen=True)
class TextSliceRoute:
    line_index: int
    segment_index: int
    bbox: XYXY
    carved: bool
    kind: str = ROUTE_SEGMENT_TEXT_OTHER

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))
        normalized = normalize_route_segment_kind(self.kind)
        if not is_text_route_segment_kind(normalized):
            raise ValueError(f"text slice route cannot use non-text kind: {normalized!r}")
        object.__setattr__(self, "kind", normalized)


@dataclass(frozen=True)
class RoutingPlan:
    lines: tuple[RoutingLine, ...]
    text_slices: tuple[TextSliceRoute, ...]
    has_layout_routes: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "text_slices", tuple(self.text_slices))
        if any(not isinstance(line, RoutingLine) for line in self.lines):
            raise TypeError("routing plan lines require typed values")
        if any(not isinstance(route, TextSliceRoute) for route in self.text_slices):
            raise TypeError("routing plan text slices require typed values")


@dataclass(frozen=True)
class BlockRoutingPlan:
    """Immutable CharOCR routes for one adopted layout block."""

    block_uid: str
    plan: RoutingPlan

    def __post_init__(self) -> None:
        if not self.block_uid:
            raise ValueError("block routing plan requires block_uid")
        if not isinstance(self.plan, RoutingPlan):
            raise TypeError("block routing plan requires a typed routing plan")


@dataclass(frozen=True)
class RouteValidationIssue:
    """A page-local routing fact that prevents CharOCR dispatch for that page."""

    code: str
    message: str
    line_index: int
    bbox: XYXY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))


@dataclass(frozen=True)
class RouteDiagnostic:
    """A non-blocking page-local routing fact retained for audit."""

    code: str
    message: str
    line_index: int
    bbox: XYXY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", xyxy(self.bbox))


@dataclass(frozen=True)
class PageRoutingPlan:
    """Immutable CharOCR dispatch input compiled from layout and PP-OCR facts.

    This object is a derived plan, not a layout/OCR source of truth.  It is
    intentionally page-scoped so an invalid route stops only that page.
    """

    page_uid: str
    routing_run_uid: str
    layout_fingerprint: str
    prepass_run_id: str
    blocks: tuple[BlockRoutingPlan, ...]
    validation_issues: tuple[RouteValidationIssue, ...] = ()
    diagnostics: tuple[RouteDiagnostic, ...] = ()
    alignments: tuple[BlockObservationAlignment, ...] = ()

    def __post_init__(self) -> None:
        if not self.page_uid:
            raise ValueError("page routing plan requires page_uid")
        if not self.routing_run_uid or not self.layout_fingerprint:
            raise ValueError("page routing plan requires routing run identity")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "validation_issues", tuple(self.validation_issues))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "alignments", tuple(self.alignments))
        if any(not isinstance(block, BlockRoutingPlan) for block in self.blocks):
            raise TypeError("page routing plan blocks require typed values")
        if any(not isinstance(issue, RouteValidationIssue) for issue in self.validation_issues):
            raise TypeError("page routing plan validation issues require typed values")
        if any(not isinstance(item, RouteDiagnostic) for item in self.diagnostics):
            raise TypeError("page routing plan diagnostics require typed values")
        if any(not isinstance(item, BlockObservationAlignment) for item in self.alignments):
            raise TypeError("page routing plan alignments require typed values")
        block_uids = [block.block_uid for block in self.blocks]
        if len(set(block_uids)) != len(block_uids):
            raise ValueError("page routing plan contains duplicate block routes")

    @property
    def is_dispatchable(self) -> bool:
        return not self.validation_issues

    def for_block(self, block_uid: str) -> RoutingPlan | None:
        for block in self.blocks:
            if block.block_uid == block_uid:
                return block.plan
        return None


def routing_line_from_record(
    index: int,
    route: dict[str, Any],
    *,
    source_field: str,
) -> RoutingLine:
    return RoutingLine(
        index=index,
        bbox=xyxy(route.get("bbox")),
        segments=tuple(
            routing_segment_from_record(segment)
            for segment in route.get("segments", [])
            if isinstance(segment, dict)
        ),
        source=str(route.get(source_field) or ""),
        text_axis=str(route.get("text_axis") or TEXT_AXIS_HORIZONTAL),
        orientation_angle=int_or_default(route.get("orientation_angle"), -1),
        ppocr_symbol_observations=tuple(
            PpOcrSymbolObservation(
                text=str(item.get("text") or ""),
                bbox=xyxy(item.get("bbox")),
                proposal_bbox=xyxy(item.get("proposal_bbox")),
                leading_space=bool(item.get("leading_space", False)),
                trailing_space=bool(item.get("trailing_space", False)),
            )
            for item in route.get("ppocr_symbol_observations", [])
            if isinstance(item, dict)
        ),
        vl_marker_observations=tuple(
            VlSemanticMarkerObservation(
                text=str(item.get("text") or ""),
                bbox=xyxy(item.get("bbox")),
                proposal_bbox=xyxy(item.get("proposal_bbox")),
            )
            for item in route.get("vl_marker_observations", [])
            if isinstance(item, dict)
        ),
    )


def routing_segment_from_record(segment: dict[str, Any]) -> RoutingSegment:
    content_bbox = segment.get("content_bbox")
    return RoutingSegment(
        kind=normalize_route_segment_kind(segment.get("kind")),
        label=str(segment.get("label") or ""),
        bbox=xyxy(segment.get("bbox")),
        text=str(segment.get("text") or ""),
        structural_block_uid=str(segment.get("structural_block_uid") or ""),
        text_source=str(segment.get("text_source") or ""),
        content_bbox=xyxy(content_bbox) if content_bbox is not None else None,
        ppocr_latin_tokens=tuple(
            PpOcrLatinTokenObservation(
                text=str(item.get("text") or ""),
                bbox=xyxy(item.get("bbox")),
            )
            for item in segment.get("ppocr_latin_tokens", [])
            if isinstance(item, dict)
        ),
    )


def routing_line_to_record(
    line: RoutingLine,
    *,
    source_field: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bbox": list(line.bbox),
        "segments": [
            {
                "kind": segment.kind,
                "label": segment.label,
                "bbox": list(segment.bbox),
                "text": segment.text,
                **(
                    {"structural_block_uid": segment.structural_block_uid}
                    if segment.structural_block_uid
                    else {}
                ),
                **({"text_source": segment.text_source} if segment.text_source else {}),
                **({"content_bbox": list(segment.content_bbox)} if segment.content_bbox else {}),
                **(
                    {
                        "ppocr_latin_tokens": [
                            {"text": token.text, "bbox": list(token.bbox)}
                            for token in segment.ppocr_latin_tokens
                        ]
                    }
                    if segment.ppocr_latin_tokens
                    else {}
                ),
            }
            for segment in line.segments
        ],
    }
    if line.source:
        record[source_field] = line.source
    if line.text_axis != TEXT_AXIS_HORIZONTAL:
        record["text_axis"] = line.text_axis
    if line.orientation_angle != -1:
        record["orientation_angle"] = line.orientation_angle
    if line.ppocr_symbol_observations:
        record["ppocr_symbol_observations"] = [
            {
                "text": observation.text,
                "bbox": list(observation.bbox),
                "proposal_bbox": list(observation.proposal_bbox),
                **({"leading_space": True} if observation.leading_space else {}),
                **({"trailing_space": True} if observation.trailing_space else {}),
            }
            for observation in line.ppocr_symbol_observations
        ]
    if line.vl_marker_observations:
        record["vl_marker_observations"] = [
            {
                "text": observation.text,
                "bbox": list(observation.bbox),
                "proposal_bbox": list(observation.proposal_bbox),
            }
            for observation in line.vl_marker_observations
        ]
    return record


def text_slice_from_record(route: dict[str, Any]) -> TextSliceRoute:
    return TextSliceRoute(
        line_index=int_or_default(route.get("line_idx"), -1),
        segment_index=int_or_default(route.get("segment_idx"), 0),
        bbox=xyxy(route.get("bbox")),
        carved=bool(route.get("carved", False)),
        kind=normalize_route_segment_kind(route.get("kind")),
    )


def text_slice_to_record(route: TextSliceRoute) -> dict[str, Any]:
    return {
        "line_idx": route.line_index,
        "segment_idx": route.segment_index,
        "bbox": list(route.bbox),
        "carved": route.carved,
        "kind": route.kind,
    }


def xyxy(value: object) -> XYXY:
    """Normalize a serialized bbox without hiding malformed geometry."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox must contain exactly four integers: {value!r}")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError(f"bbox coordinates must be integers: {value!r}")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "BlockRoutingPlan",
    "PageRoutingPlan",
    "PpOcrLatinTokenObservation",
    "PpOcrSymbolObservation",
    "VlSemanticMarkerObservation",
    "RouteDiagnostic",
    "RouteValidationIssue",
    "ROUTING_SOURCE_PPOCR_V6_PREPASS",
    "ROUTING_SOURCE_LAYOUT_VERTICAL_TEXT",
    "RoutingLine",
    "RoutingPlan",
    "RoutingSegment",
    "ROUTE_SEGMENT_FORMULA",
    "ROUTE_SEGMENT_DECORATION",
    "ROUTE_SEGMENT_SKIP",
    "ROUTE_SEGMENT_TEXT_LATIN",
    "ROUTE_SEGMENT_TEXT_OTHER",
    "TEXT_ROUTE_SEGMENT_KINDS",
    "TextSliceRoute",
    "VALID_ROUTE_SEGMENT_KINDS",
    "is_text_route_segment_kind",
    "normalize_route_segment_kind",
    "routing_line_from_record",
    "routing_line_to_record",
    "routing_segment_from_record",
    "text_slice_from_record",
    "text_slice_to_record",
    "xyxy",
]
