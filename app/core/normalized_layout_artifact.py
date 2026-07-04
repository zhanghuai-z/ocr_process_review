"""Normalized read model for external layout artifacts.

External layout engines may be PaddleOCR-VL, vector PDF extraction, or another
provider. Raw payloads remain evidence; application services should consume
this normalized view when they need layout facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.paddle_line_routing import (
    ROUTE_SUBBLOCKS_FIELD,
    block_bbox_xyxy,
    block_text,
    route_authority_label,
    route_subblocks_for_block,
)
from app.models import Page, RawOcrArtifact


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutSubregion:
    label: str
    bbox: XYXY
    text: str = ""
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class LayoutRegion:
    index: int
    label: str
    bbox: XYXY
    text: str = ""
    confidence: float | None = None
    raw: dict[str, Any] | None = None
    subregions: tuple[LayoutSubregion, ...] = ()


@dataclass(frozen=True)
class NormalizedLayoutArtifact:
    artifact_uid: str
    page_uid: str
    engine: str
    engine_version: str
    page_width: int
    page_height: int
    regions: tuple[LayoutRegion, ...]


def normalized_layout_artifact_from_page(page: Page) -> NormalizedLayoutArtifact:
    artifact = page.raw_layout_artifact
    if artifact is None:
        return NormalizedLayoutArtifact(
            artifact_uid="",
            page_uid=page.uid,
            engine="",
            engine_version="",
            page_width=page.width,
            page_height=page.height,
            regions=(),
        )
    return normalized_layout_artifact_from_raw(page, artifact)


def normalized_layout_artifact_from_raw(
    page: Page,
    artifact: RawOcrArtifact,
) -> NormalizedLayoutArtifact:
    regions: list[LayoutRegion] = []
    route_attachments = {
        int(index): [dict(item) for item in values]
        for index, values in dict(artifact.route_attachments or {}).items()
    }
    for index, record in enumerate(artifact.records):
        if not isinstance(record, dict):
            continue
        raw = dict(record)
        route_record = _route_record_from_raw(raw, route_attachments.get(index))
        subregions = tuple(
            LayoutSubregion(
                label=str(subregion["label"]),
                bbox=tuple(subregion["bbox"]),
                text=str(subregion.get("text") or ""),
                raw=dict(subregion.get("raw") or {}),
            )
            for subregion in route_subblocks_for_block(route_record, page.width, page.height)
        )
        regions.append(
            LayoutRegion(
                index=index,
                label=route_authority_label(raw),
                bbox=block_bbox_xyxy(raw, page.width, page.height),
                text=block_text(raw),
                confidence=_record_confidence(raw),
                raw=raw,
                subregions=subregions,
            )
        )
    return NormalizedLayoutArtifact(
        artifact_uid=artifact.uid,
        page_uid=artifact.page_uid or page.uid,
        engine=artifact.engine,
        engine_version=artifact.engine_version,
        page_width=page.width,
        page_height=page.height,
        regions=tuple(regions),
    )


def normalized_layout_regions(page: Page) -> tuple[LayoutRegion, ...]:
    return normalized_layout_artifact_from_page(page).regions


def layout_region_route_record(region: LayoutRegion) -> dict[str, Any]:
    """Return a transient routing input record for existing route compilers."""
    record = dict(region.raw or {})
    if region.subregions:
        record[ROUTE_SUBBLOCKS_FIELD] = [
            dict(subregion.raw or {
                "block_label": subregion.label,
                "block_bbox": list(subregion.bbox),
                "block_content": subregion.text,
            })
            for subregion in region.subregions
        ]
    return record


def _route_record_from_raw(
    raw: dict[str, Any],
    attachments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not attachments:
        return raw
    route_record = dict(raw)
    route_record[ROUTE_SUBBLOCKS_FIELD] = [dict(item) for item in attachments]
    return route_record


def _record_confidence(record: dict[str, Any]) -> float | None:
    try:
        value = record.get("score")
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


__all__ = [
    "LayoutRegion",
    "LayoutSubregion",
    "NormalizedLayoutArtifact",
    "layout_region_route_record",
    "normalized_layout_artifact_from_page",
    "normalized_layout_artifact_from_raw",
    "normalized_layout_regions",
]
