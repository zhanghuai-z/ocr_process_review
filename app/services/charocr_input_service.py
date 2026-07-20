"""Compile the adopted layout into the sole CharOCR native input."""
from __future__ import annotations

import json

from app.core.paddle_response import parsing_records_from_item, result_items
from app.models.charocr_execution import CharOcrInputRow, CharOcrPageRequest
from app.models.charocr_routing import PageRoutingPlan
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import PageRecord


def _artifact_records(artifact: PaddleArtifact) -> tuple[dict[str, object], ...]:
    payload = json.loads(artifact.payload_json)
    if not isinstance(payload, dict):
        raise ValueError("layout Paddle artifact must contain an object response")
    records: list[dict[str, object]] = []
    for item in result_items(payload, "layoutParsingResults"):
        records.extend(dict(record) for record in parsing_records_from_item(item))
    return tuple(records)


def _origin_content(
    block,
    records: tuple[dict[str, object], ...],
) -> str:
    origin = block.origin
    index = origin.raw_index
    if index is None or index < 0 or index >= len(records):
        return ""
    value = records[index].get("block_content")
    return value.strip() if isinstance(value, str) else ""


def compile_charocr_page_request(
    *,
    project_uid: str,
    page: PageRecord,
    layout: LayoutSnapshot,
    artifact: PaddleArtifact,
    routing_plan: PageRoutingPlan,
) -> CharOcrPageRequest:
    """Compile rows without consulting a projection, cache, or proof model."""
    if page.project_uid != project_uid or artifact.project_uid != project_uid:
        raise ValueError("CharOCR inputs belong to different projects")
    if layout.page_uid != page.uid or artifact.page_uid != page.uid:
        raise ValueError("CharOCR inputs belong to different pages")
    if layout.artifact_uid != artifact.uid:
        raise ValueError("layout does not reference the supplied Paddle artifact")
    records = _artifact_records(artifact)
    rows = tuple(
        CharOcrInputRow(
            block_uid=block.uid,
            label=block.source_label or block.block_type.value,
            bbox=block.bbox.to_xyxy(),
            content=_origin_content(block, records),
            ocr_policy=block.ocr_policy,
            authorship=block.authorship,
            order=block.order,
        )
        for block in layout.blocks
    )
    return CharOcrPageRequest(
        project_uid=project_uid,
        page=page,
        layout=layout,
        routing_plan=routing_plan,
        rows=rows,
    )


__all__ = ["compile_charocr_page_request"]
