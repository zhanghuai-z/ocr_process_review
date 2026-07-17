"""Build persistable audit drafts from one compiled routing run."""
from __future__ import annotations

from app.models.charocr_routing import PageRoutingPlan
from app.models.ocr_routing_observation import RoutingObservationBundle
from app.models.ocr_routing_run_audit import (
    BlockAlignmentAuditSummary,
    BlockVlObservationSummary,
    OcrRoutingRunAuditDraft,
    RouteAuditSummary,
    RoutingAuditMessage,
)


def build_ocr_routing_run_audit_draft(
    observations: RoutingObservationBundle[object],
    plan: PageRoutingPlan,
) -> OcrRoutingRunAuditDraft:
    if observations.run_uid != plan.routing_run_uid:
        raise ValueError("routing audit plan belongs to another observation run")
    if observations.layout_fingerprint != plan.layout_fingerprint:
        raise ValueError("routing audit plan belongs to another layout scope")
    lines = [line for block in plan.blocks for line in block.plan.lines]
    return OcrRoutingRunAuditDraft(
        run_uid=plan.routing_run_uid,
        page_uid=plan.page_uid,
        image_hash=observations.image_hash,
        layout_fingerprint=plan.layout_fingerprint,
        pp_run_id=plan.prepass_run_id,
        block_vl_observations=tuple(
            BlockVlObservationSummary(
                block_uid=item.block_uid,
                status=item.status.value,
                observation_uid=item.uid,
                text_length=len(item.text),
                record_count=len(item.regions),
                raw_refs=tuple(ref for ref in (item.raw_response_ref,) if ref),
            )
            for item in observations.block_vl_observations
        ),
        alignment_statuses=tuple(
            BlockAlignmentAuditSummary(
                block_uid=item.block_uid,
                status=item.status.value,
                matched_span_count=len(item.pp_source_indices),
                message=item.detail,
            )
            for item in plan.alignments
        ),
        route_summary=RouteAuditSummary(
            block_count=len(plan.blocks),
            line_count=len(lines),
            segment_count=sum(len(line.segments) for line in lines),
            is_dispatchable=plan.is_dispatchable,
            validation_issues=tuple(
                RoutingAuditMessage(
                    code=item.code,
                    message=item.message,
                    line_index=item.line_index,
                    bbox=item.bbox,
                )
                for item in plan.validation_issues
            ),
            diagnostics=tuple(
                RoutingAuditMessage(
                    code=item.code,
                    message=item.message,
                    line_index=item.line_index,
                    bbox=item.bbox,
                )
                for item in plan.diagnostics
            ),
        ),
    )


__all__ = ["build_ocr_routing_run_audit_draft"]
