"""Validate that automatic layout truth represents Paddle inline formulas."""
from __future__ import annotations

import json

from app.adapters.paddle.inline_formula_observations import (
    inline_formula_detector_observations,
)
from app.models.enums import BlockSource, BlockType
from app.models.geometry import BBox
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact


class StaleInlineFormulaLayoutError(RuntimeError):
    """An old automatic layout omitted detector formula geometry."""


def require_current_automatic_inline_formula_layout(
    snapshot: LayoutSnapshot,
    artifact: PaddleArtifact,
    *,
    page_width: int,
    page_height: int,
) -> None:
    """Fail old automatic snapshots without mutating edited layout truth."""
    if snapshot.source_engine == "layout_edit" or any(
        block.authorship is not BlockSource.AUTO_LAYOUT for block in snapshot.blocks
    ):
        return
    payload = json.loads(artifact.payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Paddle layout artifact must contain an object response")
    observations = inline_formula_detector_observations(
        payload,
        page_width=page_width,
        page_height=page_height,
    )
    equation_bboxes = [
        block.bbox for block in snapshot.blocks if block.block_type is BlockType.EQUATION
    ]
    missing = [
        observation
        for observation in observations
        if not any(_contains(bbox, observation.bbox) for bbox in equation_bboxes)
    ]
    if missing:
        raise StaleInlineFormulaLayoutError(
            "当前自动版面缺少 "
            f"{len(missing)} 个 Paddle 行内公式框，请先重新运行该页版面分析再执行 OCR"
        )


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


__all__ = [
    "StaleInlineFormulaLayoutError",
    "require_current_automatic_inline_formula_layout",
]
