"""Acquire versioned VL/PP observations before compiling CharOCR routes."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.core.layout_scope import layout_snapshot_fingerprint, page_image_hash
from app.core.normalized_layout_artifact import normalized_layout_artifact_from_page
from app.core.paddle_layout_schema import normalize_paddle_layout_record
from app.core.paddle_response import parsing_records_from_item, result_items
from app.core.paddle_v16_client import PaddleV16LayoutClient, build_paddle_v16_optional_payload
from app.models import BlockSource, BlockType, OcrPolicy, Page
from app.models.entity_id import ensure_entity_uid
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.ocr_routing_observation import (
    BlockVlObservation,
    BlockVlObservationStatus,
    BlockVlTextRegion,
    RoutingObservationBundle,
)


TEXT_EXCLUDED_TYPES = frozenset({
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.UNKNOWN,
})


class BlockVlObservationRefreshError(RuntimeError):
    """Raised when a required fresh VL request cannot produce an observation."""


def acquire_routing_observation_bundle(
    *,
    page: Page,
    snapshot: LayoutSnapshot,
    image_bgr: np.ndarray,
    prepass: object,
    vl_client: PaddleV16LayoutClient,
) -> RoutingObservationBundle[object]:
    """Acquire all external observations against one immutable layout scope."""
    if snapshot.page_uid != page.uid:
        raise ValueError("routing observation snapshot belongs to another page")
    image_hash = page_image_hash(image_bgr)
    layout_fingerprint = layout_snapshot_fingerprint(snapshot)
    run_uid = ensure_entity_uid("", "ocrrun")
    normalized_original = normalized_layout_artifact_from_page(page)
    observations: list[BlockVlObservation] = []
    for block in snapshot.blocks:
        if not _requires_text_observation(block):
            continue
        original = _observation_from_original_artifact(
            page=page,
            block=block,
            image_hash=image_hash,
            layout_fingerprint=layout_fingerprint,
            normalized_original=normalized_original,
        )
        if original is not None:
            observations.append(original)
            continue
        observations.append(_refresh_block_observation(
            page=page,
            block=block,
            image_bgr=image_bgr,
            image_hash=image_hash,
            layout_fingerprint=layout_fingerprint,
            client=vl_client,
            batch_id=f"{run_uid}-{block.uid}",
        ))
    return RoutingObservationBundle(
        run_uid=run_uid,
        snapshot=snapshot,
        prepass=prepass,
        image_hash=image_hash,
        layout_fingerprint=layout_fingerprint,
        block_vl_observations=tuple(observations),
    )


def _requires_text_observation(block: LayoutBlockSnapshot) -> bool:
    return block.ocr_policy == OcrPolicy.TEXT_OCR and block.block_type not in TEXT_EXCLUDED_TYPES


def _observation_from_original_artifact(
    *,
    page: Page,
    block: LayoutBlockSnapshot,
    image_hash: str,
    layout_fingerprint: str,
    normalized_original: Any,
) -> BlockVlObservation | None:
    origin = block.origin
    if (
        origin is None
        or origin.created_by != BlockSource.AUTO_LAYOUT.value
        or origin.raw_artifact_uid != normalized_original.artifact_uid
        or origin.raw_index is None
        or origin.original_bbox is None
        or origin.original_bbox.to_xyxy() != block.bbox.to_xyxy()
        or origin.original_kind != block.block_type
        or str(origin.source_label or "") != str(block.source_label or "")
    ):
        return None
    region = next(
        (candidate for candidate in normalized_original.regions if candidate.index == origin.raw_index),
        None,
    )
    if region is None or region.bbox != block.bbox.to_xyxy():
        return None
    text_regions = () if not region.text else (
        BlockVlTextRegion(
            index=region.index,
            label=region.label,
            text=region.text,
            bbox=region.bbox,
        ),
    )
    return BlockVlObservation(
        page_uid=page.uid,
        block_uid=block.uid,
        block_bbox=block.bbox.to_xyxy(),
        image_hash=image_hash,
        layout_fingerprint=layout_fingerprint,
        status=(
            BlockVlObservationStatus.OBSERVED
            if text_regions
            else BlockVlObservationStatus.EMPTY
        ),
        regions=text_regions,
        source_artifact_uid=normalized_original.artifact_uid,
        source_run_id=origin.source_run_id,
        raw_response_ref=normalized_original.artifact_uid,
        attempts=0,
    )


def _refresh_block_observation(
    *,
    page: Page,
    block: LayoutBlockSnapshot,
    image_bgr: np.ndarray,
    image_hash: str,
    layout_fingerprint: str,
    client: PaddleV16LayoutClient,
    batch_id: str,
) -> BlockVlObservation:
    bbox = block.bbox.clamp(image_bgr.shape[1], image_bgr.shape[0])
    if bbox.area <= 0:
        raise BlockVlObservationRefreshError(
            f"current layout block has an invalid VL observation crop: {block.uid}"
        )
    crop = image_bgr[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
    response: dict[str, Any] | None = None
    regions: tuple[BlockVlTextRegion, ...] | None = None
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(2):
        attempts = attempt + 1
        try:
            response = client.analyze_image(
                crop,
                optional_payload=build_paddle_v16_optional_payload(),
                batch_id=f"{batch_id}-{attempt + 1}",
            )
            regions = _validated_text_regions_from_crop_response(
                response,
                bbox.to_xyxy(),
                crop.shape[1],
                crop.shape[0],
            )
            break
        except Exception as exc:
            last_error = exc
    if response is None or regions is None:
        raise BlockVlObservationRefreshError(
            f"VL1.6 block observation failed after retry: block={block.uid}: {last_error}"
        ) from last_error

    job = response.get("paddle_v16", {}) if isinstance(response, dict) else {}
    raw_response_ref = str(job.get("jobId") or "") if isinstance(job, dict) else ""
    return BlockVlObservation(
        page_uid=page.uid,
        block_uid=block.uid,
        block_bbox=block.bbox.to_xyxy(),
        image_hash=image_hash,
        layout_fingerprint=layout_fingerprint,
        status=(BlockVlObservationStatus.OBSERVED if regions else BlockVlObservationStatus.EMPTY),
        regions=regions,
        source_run_id=raw_response_ref,
        raw_response_ref=raw_response_ref,
        attempts=attempts,
    )


def _validated_text_regions_from_crop_response(
    response: dict[str, Any],
    crop_bbox: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
) -> tuple[BlockVlTextRegion, ...]:
    if not isinstance(response, dict):
        raise TypeError("VL1.6 block response must be an object")
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("layoutParsingResults"), list):
        raise ValueError("VL1.6 block response lacks layoutParsingResults")
    x_offset, y_offset = crop_bbox[0], crop_bbox[1]
    regions: list[BlockVlTextRegion] = []
    for item in result_items(response, "layoutParsingResults"):
        for record in parsing_records_from_item(item):
            normalized = normalize_paddle_layout_record(
                record,
                page_width=crop_width,
                page_height=crop_height,
            )
            if normalized is None or not normalized.text:
                continue
            local_bbox = normalized.bbox.to_xyxy()
            regions.append(BlockVlTextRegion(
                index=len(regions),
                label=normalized.label,
                text=normalized.text,
                bbox=(
                    local_bbox[0] + x_offset,
                    local_bbox[1] + y_offset,
                    local_bbox[2] + x_offset,
                    local_bbox[3] + y_offset,
                ),
            ))
    return tuple(regions)


__all__ = [
    "BlockVlObservationRefreshError",
    "acquire_routing_observation_bundle",
]
