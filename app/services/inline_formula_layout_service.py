"""Derive inline-formula layout edits from immutable Paddle artifacts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.core.text_classification import is_formula_marker_token
from app.core.paddle_labels import normalize_paddle_label
from app.core.paddle_line_routing import block_text, route_subblocks_for_block
from app.core.paddle_response import parsing_records_from_item, result_items
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.services.layout_edit_service import (
    LayoutEditCommand,
    LayoutEditResult,
    LayoutEditService,
)


INLINE_FORMULA_SPAN_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class InlineFormulaRegion:
    """One formula geometry fact addressed within a Paddle artifact."""

    artifact_uid: str
    parent_index: int
    subregion_index: int
    parent_label: str
    label: str
    bbox: BBox
    text: str = ""


class InlineFormulaLayoutService:
    """Produce pure snapshot edits; persistence belongs to the caller."""

    def iter_regions(
        self,
        artifact: PaddleArtifact,
        *,
        page_width: int,
        page_height: int,
    ) -> tuple[InlineFormulaRegion, ...]:
        if not isinstance(artifact, PaddleArtifact):
            raise TypeError("inline formula regions require PaddleArtifact")
        _validate_dimensions(page_width, page_height)
        records = _artifact_layout_records(artifact)
        regions: list[InlineFormulaRegion] = []
        for parent_index, record in enumerate(records):
            parent_label = str(record.get("block_label") or record.get("label") or "")
            subregions = route_subblocks_for_block(record, page_width, page_height)
            formula_subregions = [
                (index, value)
                for index, value in enumerate(subregions)
                if normalize_paddle_label(value.get("label")) == "inline_formula"
            ]
            marker_spans = _formula_spans(block_text(record))
            for formula_index, (subregion_index, value) in enumerate(formula_subregions):
                bbox = BBox.from_xyxy(*value["bbox"]).clamp(page_width, page_height)
                if bbox.w <= 0 or bbox.h <= 0:
                    continue
                text = str(value.get("text") or "")
                if not text and formula_index < len(marker_spans):
                    text = marker_spans[formula_index]
                if text and is_formula_marker_token(text):
                    continue
                regions.append(
                    InlineFormulaRegion(
                        artifact_uid=artifact.uid,
                        parent_index=parent_index,
                        subregion_index=subregion_index,
                        parent_label=parent_label,
                        label=str(value.get("label") or "inline_formula"),
                        bbox=bbox,
                        text=text,
                    )
                )
        return tuple(regions)

    def snapshot_edit_commands(
        self,
        snapshot: LayoutSnapshot,
        artifact: PaddleArtifact,
        *,
        page_width: int,
        page_height: int,
        handled_bboxes: Iterable[tuple[int, int, int, int]] = (),
    ) -> tuple[LayoutEditCommand, ...]:
        """Return sequential edit intents without changing ``snapshot``."""

        _validate_snapshot_artifact(snapshot, artifact)
        handled = {tuple(int(item) for item in bbox) for bbox in handled_bboxes}
        existing = {
            block.bbox.to_xyxy()
            for block in snapshot.blocks
            if normalize_paddle_label(block.source_label) == "inline_formula"
        }
        existing.update(
            block.origin.original_bbox.to_xyxy()
            for block in snapshot.blocks
            if block.origin.raw_artifact_uid == artifact.uid
            and block.origin.original_bbox is not None
        )
        commands: list[LayoutEditCommand] = []
        next_revision = snapshot.revision
        for region in self.iter_regions(
            artifact,
            page_width=page_width,
            page_height=page_height,
        ):
            bbox = region.bbox.to_xyxy()
            if bbox in handled or bbox in existing:
                continue
            commands.append(
                LayoutEditCommand.create_block(
                    snapshot.page_uid,
                    next_revision,
                    region.bbox,
                    BlockType.EQUATION,
                    normalize_paddle_label(region.label) or "inline_formula",
                    new_block_uid=(
                        f"inline_formula_{artifact.uid}_"
                        f"{region.parent_index}_{region.subregion_index}"
                    ),
                )
            )
            next_revision += 1
            existing.add(bbox)
        return tuple(commands)

    def apply_snapshot_edits(
        self,
        snapshot: LayoutSnapshot,
        artifact: PaddleArtifact,
        *,
        page_width: int,
        page_height: int,
        handled_bboxes: Iterable[tuple[int, int, int, int]] = (),
    ) -> tuple[LayoutEditResult, ...]:
        """Apply generated commands through ``LayoutEditService`` in memory."""

        commands = self.snapshot_edit_commands(
            snapshot,
            artifact,
            page_width=page_width,
            page_height=page_height,
            handled_bboxes=handled_bboxes,
        )
        editor = LayoutEditService()
        current = snapshot
        results: list[LayoutEditResult] = []
        regions_by_bbox = {
            region.bbox.to_xyxy(): region
            for region in self.iter_regions(
                artifact,
                page_width=page_width,
                page_height=page_height,
            )
        }
        for command in commands:
            result = editor.apply(current, command)
            created_uid = result.affected_block_uids[0]
            created = next(block for block in result.snapshot.blocks if block.uid == created_uid)
            region = regions_by_bbox[created.bbox.to_xyxy()]
            enriched = replace(
                created,
                authorship=BlockSource.AUTO_LAYOUT,
                ocr_policy=OcrPolicy.PRESERVE_AS_FORMULA,
                origin=BlockOrigin(
                    created_by=BlockSource.AUTO_LAYOUT.value,
                    source_engine=artifact.source_engine,
                    source_run_id=artifact.source_run_id,
                    vendor_label=region.label,
                    original_bbox=region.bbox,
                    original_kind=BlockType.EQUATION,
                    raw_artifact_uid=artifact.uid,
                    raw_index=region.parent_index,
                ),
            )
            next_snapshot = replace(
                result.snapshot,
                source_engine=artifact.source_engine,
                source_run_id=artifact.source_run_id,
                blocks=tuple(
                    enriched if block.uid == created_uid else block
                    for block in result.snapshot.blocks
                ),
            )
            result = replace(result, snapshot=next_snapshot)
            results.append(result)
            current = next_snapshot
        return tuple(results)


def _validate_dimensions(page_width: int, page_height: int) -> None:
    if isinstance(page_width, bool) or not isinstance(page_width, int) or page_width <= 0:
        raise ValueError("page_width must be positive")
    if isinstance(page_height, bool) or not isinstance(page_height, int) or page_height <= 0:
        raise ValueError("page_height must be positive")


def _validate_snapshot_artifact(
    snapshot: LayoutSnapshot,
    artifact: PaddleArtifact,
) -> None:
    if not isinstance(snapshot, LayoutSnapshot):
        raise TypeError("inline formula edits require LayoutSnapshot")
    if not isinstance(artifact, PaddleArtifact):
        raise TypeError("inline formula edits require PaddleArtifact")
    if snapshot.page_uid != artifact.page_uid:
        raise ValueError("layout snapshot and Paddle artifact belong to different pages")
    if snapshot.artifact_uid != artifact.uid:
        raise ValueError("layout snapshot does not reference the supplied Paddle artifact")


def _artifact_layout_records(artifact: PaddleArtifact) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(artifact.payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Paddle artifact payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Paddle layout artifact payload must be an object")
    records: list[dict[str, Any]] = []
    for item in result_items(payload, "layoutParsingResults"):
        records.extend(dict(record) for record in parsing_records_from_item(item))
    return tuple(records)


def _formula_spans(text: str) -> list[str]:
    return [match.group(0) for match in INLINE_FORMULA_SPAN_RE.finditer(text or "")]


__all__ = [
    "InlineFormulaLayoutService",
    "InlineFormulaRegion",
]
