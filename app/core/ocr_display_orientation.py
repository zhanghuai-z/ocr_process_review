"""Derive an auditable page display orientation from OCR routing observations."""
from __future__ import annotations

from dataclasses import dataclass
import json

from app.models.charocr_routing import PageRoutingPlan


DISPLAY_ROTATION_KEY = "display_rotation_quarters_clockwise"
DISPLAY_ROTATION_SOURCE_KEY = "display_rotation_source"
DISPLAY_ROTATION_EVIDENCE_KEY = "display_rotation_evidence"
DISPLAY_ROTATION_SOURCE = "ppocrv6:textline_orientation_consensus"


@dataclass(frozen=True, slots=True)
class OcrDisplayOrientation:
    quarters_clockwise: int
    weights: tuple[int, int, int, int]
    confidence: float

    def metadata(self) -> tuple[tuple[str, str], ...]:
        evidence = json.dumps(
            {"weights": list(self.weights), "confidence": round(self.confidence, 6)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            (DISPLAY_ROTATION_KEY, str(self.quarters_clockwise)),
            (DISPLAY_ROTATION_SOURCE_KEY, DISPLAY_ROTATION_SOURCE),
            (DISPLAY_ROTATION_EVIDENCE_KEY, evidence),
        )


def observe_page_display_orientation(plan: PageRoutingPlan) -> OcrDisplayOrientation:
    """Choose a page display rotation only from dispatchable oriented text lines."""
    if not isinstance(plan, PageRoutingPlan):
        raise TypeError("display orientation requires PageRoutingPlan")
    weights = [0, 0, 0, 0]
    for line in (
        line
        for block in plan.blocks
        for line in block.plan.lines
    ):
        if not any(
            segment.kind in {"text_other", "text_latin"}
            for segment in line.segments
        ):
            continue
        width = max(1, line.bbox[2] - line.bbox[0])
        height = max(1, line.bbox[3] - line.bbox[1])
        if line.text_axis == "vertical":
            if line.orientation_angle not in {0, 180}:
                continue
            rotation = 3 if line.orientation_angle == 0 else 1
            weight = max(1, round(height / width))
        else:
            rotation = 2 if line.orientation_angle == 180 else 0
            weight = max(1, round(width / height))
        weights[rotation] += min(weight, 80)
    total = sum(weights)
    if total == 0:
        return OcrDisplayOrientation(0, tuple(weights), 0.0)
    winner = max(range(4), key=lambda index: weights[index])
    confidence = weights[winner] / total
    if winner != 0 and (weights[winner] < 4 or confidence < 0.65):
        winner = 0
    return OcrDisplayOrientation(winner, tuple(weights), confidence)


def display_rotation_from_metadata(metadata: tuple[tuple[str, str], ...]) -> int:
    raw = dict(metadata).get(DISPLAY_ROTATION_KEY, "0")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value in {0, 1, 2, 3} else 0


__all__ = [
    "DISPLAY_ROTATION_EVIDENCE_KEY",
    "DISPLAY_ROTATION_KEY",
    "DISPLAY_ROTATION_SOURCE",
    "DISPLAY_ROTATION_SOURCE_KEY",
    "OcrDisplayOrientation",
    "display_rotation_from_metadata",
    "observe_page_display_orientation",
]
