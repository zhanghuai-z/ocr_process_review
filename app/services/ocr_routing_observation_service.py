"""Acquire versioned VL/PP observations before compiling CharOCR routes."""
from __future__ import annotations

from typing import Any
import json

import numpy as np

from app.core.layout_scope import layout_snapshot_fingerprint, page_image_hash
from app.adapters.paddle.inline_formula_observations import (
    PaddleInlineFormulaDetectorObservation,
    inline_formula_detector_observations,
)
from app.core.paddle_layout_schema import normalize_paddle_layout_record
from app.core.paddle_response import parsing_records_from_item, result_items
from app.core.paddle_v16_client import PaddleV16LayoutClient, build_paddle_v16_optional_payload
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.entity_id import ensure_entity_uid
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord
from app.models.ocr_routing_observation import (
    BlockVlObservation,
    BlockVlObservationStatus,
    BlockVlTextRegion,
    InlineFormulaTextObservation,
    InlineFormulaTextObservationStatus,
    RoutingObservationBundle,
)
from app.services.formula_crop_ocr_service import recognize_formula_bboxes_with_retry


TEXT_EXCLUDED_TYPES = frozenset({
    BlockType.EQUATION,
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.UNKNOWN,
})
INLINE_FORMULA_PARENT_SOURCE = "paddlevl:inline_formula_parent_span"
INLINE_FORMULA_CROP_SOURCE = "paddlevl:inline_formula_crop_ocr"
INLINE_FORMULA_UNRESOLVED_SOURCE = "paddlevl:inline_formula_unresolved"


class BlockVlObservationRefreshError(RuntimeError):
    """Raised when a required fresh VL request cannot produce an observation."""


def acquire_routing_observation_bundle(
    *,
    page: PageRecord,
    snapshot: LayoutSnapshot,
    artifact: PaddleArtifact,
    image_bgr: np.ndarray,
    prepass: object,
    vl_client: PaddleV16LayoutClient,
) -> RoutingObservationBundle[object]:
    """Acquire all external observations against one immutable layout scope."""
    if snapshot.page_uid != page.uid or artifact.page_uid != page.uid:
        raise ValueError("routing observations belong to different pages")
    if artifact.project_uid != page.project_uid:
        raise ValueError("routing artifact belongs to another project")
    if snapshot.artifact_uid != artifact.uid:
        raise ValueError("routing snapshot does not reference the supplied Paddle artifact")
    image_hash = page_image_hash(image_bgr)
    layout_fingerprint = layout_snapshot_fingerprint(snapshot)
    run_uid = ensure_entity_uid("", "ocrrun")
    original_records = _artifact_layout_records(artifact)
    detector_observations = _artifact_inline_formula_observations(
        artifact,
        page_width=page.width,
        page_height=page.height,
    )
    observations: list[BlockVlObservation] = []
    for block in snapshot.blocks:
        if not _requires_text_observation(block):
            continue
        original = _observation_from_original_artifact(
            page=page,
            block=block,
            artifact=artifact,
            image_hash=image_hash,
            layout_fingerprint=layout_fingerprint,
            original_records=original_records,
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
    formula_observations = _inline_formula_text_observations(
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        image_bgr=image_bgr,
        client=vl_client,
        run_uid=run_uid,
        detector_observations=detector_observations,
    )
    return RoutingObservationBundle(
        run_uid=run_uid,
        snapshot=snapshot,
        prepass=prepass,
        image_hash=image_hash,
        layout_fingerprint=layout_fingerprint,
        block_vl_observations=tuple(observations),
        inline_formula_observations=formula_observations,
    )


def _inline_formula_text_observations(
    *,
    page: PageRecord,
    snapshot: LayoutSnapshot,
    artifact: PaddleArtifact,
    image_bgr: np.ndarray,
    client: PaddleV16LayoutClient,
    run_uid: str,
    detector_observations: tuple[PaddleInlineFormulaDetectorObservation, ...],
) -> tuple[InlineFormulaTextObservation, ...]:
    blocks = [
        block
        for block in snapshot.blocks
        if block.block_type is BlockType.EQUATION
        and str(block.source_label or "").strip().lower() == "inline_formula"
    ]
    if not blocks:
        return ()
    matched: dict[str, PaddleInlineFormulaDetectorObservation] = {}
    for block in blocks:
        by_path = [
            item
            for item in detector_observations
            if block.origin.raw_json_path
            and item.raw_json_path == block.origin.raw_json_path
        ]
        candidates = by_path or [
            item for item in detector_observations if item.bbox == block.bbox
        ]
        if len(candidates) == 1:
            matched[block.uid] = candidates[0]

    result_by_uid: dict[str, InlineFormulaTextObservation] = {}
    unresolved_blocks: list[LayoutBlockSnapshot] = []
    for block in blocks:
        detector = matched.get(block.uid)
        if detector is not None and detector.exact_parent_text:
            result_by_uid[block.uid] = InlineFormulaTextObservation(
                page_uid=page.uid,
                block_uid=block.uid,
                bbox=block.bbox.to_xyxy(),
                status=InlineFormulaTextObservationStatus.OBSERVED,
                text=detector.exact_parent_text,
                source=INLINE_FORMULA_PARENT_SOURCE,
                source_artifact_uid=artifact.uid,
                raw_response_ref=detector.raw_json_path,
            )
        else:
            unresolved_blocks.append(block)

    outcome = None
    if unresolved_blocks:
        outcome = recognize_formula_bboxes_with_retry(
            image_bgr,
            [block.bbox.to_xyxy() for block in unresolved_blocks],
            client=client,
            batch_id_prefix=f"{run_uid}-inline-formula",
        )
    for index, block in enumerate(unresolved_blocks):
        crop_text = _canonical_inline_formula_text(
            outcome.texts_by_index.get(index, "") if outcome is not None else ""
        )
        if crop_text:
            result_by_uid[block.uid] = InlineFormulaTextObservation(
                page_uid=page.uid,
                block_uid=block.uid,
                bbox=block.bbox.to_xyxy(),
                status=InlineFormulaTextObservationStatus.OBSERVED,
                text=crop_text,
                source=INLINE_FORMULA_CROP_SOURCE,
                source_artifact_uid=artifact.uid,
                raw_response_ref=f"formula-crop:{index}",
                attempts=outcome.attempts,
            )
        else:
            result_by_uid[block.uid] = InlineFormulaTextObservation(
                page_uid=page.uid,
                block_uid=block.uid,
                bbox=block.bbox.to_xyxy(),
                status=InlineFormulaTextObservationStatus.UNRESOLVED,
                text="",
                source=INLINE_FORMULA_UNRESOLVED_SOURCE,
                source_artifact_uid=artifact.uid,
                raw_response_ref=f"formula-crop:{index}",
                attempts=outcome.attempts if outcome is not None else 0,
                error=outcome.error if outcome is not None else "formula crop was not attempted",
            )
    return tuple(result_by_uid[block.uid] for block in blocks)


def _artifact_inline_formula_observations(
    artifact: PaddleArtifact,
    *,
    page_width: int,
    page_height: int,
) -> tuple[PaddleInlineFormulaDetectorObservation, ...]:
    payload = json.loads(artifact.payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Paddle layout artifact must contain an object response")
    return inline_formula_detector_observations(
        payload,
        page_width=page_width,
        page_height=page_height,
    )


def _canonical_inline_formula_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    pairs = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    for left, right in pairs:
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left):len(value) - len(right)].strip()
            break
    return f"${value}$" if value else ""


def _requires_text_observation(block: LayoutBlockSnapshot) -> bool:
    return block.ocr_policy == OcrPolicy.TEXT_OCR and block.block_type not in TEXT_EXCLUDED_TYPES


def _observation_from_original_artifact(
    *,
    page: PageRecord,
    block: LayoutBlockSnapshot,
    artifact: PaddleArtifact,
    image_hash: str,
    layout_fingerprint: str,
    original_records: tuple[dict[str, Any], ...],
) -> BlockVlObservation | None:
    origin = block.origin
    if (
        origin is None
        or block.authorship != BlockSource.AUTO_LAYOUT
        or origin.created_by != BlockSource.AUTO_LAYOUT.value
        or origin.raw_artifact_uid != artifact.uid
        or origin.raw_index is None
        or origin.original_bbox is None
        or origin.original_bbox.to_xyxy() != block.bbox.to_xyxy()
        or origin.original_kind != block.block_type
    ):
        return None
    if origin.raw_index >= len(original_records):
        return None
    region = normalize_paddle_layout_record(
        original_records[origin.raw_index],
        page_width=page.width,
        page_height=page.height,
    )
    if region is None or region.bbox.to_xyxy() != block.bbox.to_xyxy():
        return None
    text_regions = () if not region.text else (
        BlockVlTextRegion(
            index=origin.raw_index,
            label=region.label,
            text=region.text,
            bbox=region.bbox.to_xyxy(),
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
        source_artifact_uid=artifact.uid,
        source_run_id=origin.source_run_id,
        raw_response_ref=artifact.uid,
        attempts=0,
    )


def _refresh_block_observation(
    *,
    page: PageRecord,
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
    errors: list[Exception] = []
    attempts = 0
    for attempt in range(2):
        attempts = attempt + 1
        try:
            response = client.analyze_image(
                crop,
                optional_payload=build_paddle_v16_optional_payload(),
                batch_id=f"{batch_id}-1" if attempt == 0 else "",
            )
            regions = _validated_text_regions_from_crop_response(
                response,
                bbox.to_xyxy(),
                crop.shape[1],
                crop.shape[0],
            )
            break
        except Exception as exc:
            errors.append(exc)
    if response is None or regions is None:
        details = "; ".join(
            f"attempt {index}: {type(error).__name__}: {error}"
            for index, error in enumerate(errors, start=1)
        ) or "no response"
        raise BlockVlObservationRefreshError(
            "VL1.6 block observation failed after retry: "
            f"page={page.source_path!r} block={block.uid} "
            f"label={block.source_label!r} bbox={bbox.to_xyxy()} "
            f"authorship={block.authorship.value}: {details}"
        ) from (errors[-1] if errors else None)

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


def _artifact_layout_records(artifact: PaddleArtifact) -> tuple[dict[str, Any], ...]:
    import json

    payload = json.loads(artifact.payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Paddle layout artifact must contain an object response")
    records: list[dict[str, Any]] = []
    for item in result_items(payload, "layoutParsingResults"):
        records.extend(dict(record) for record in parsing_records_from_item(item))
    return tuple(records)


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
