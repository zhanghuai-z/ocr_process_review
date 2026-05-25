"""Project-model to Export IR v1 builder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.export.ir import (
    EXPORT_IR_VERSION,
    ExportAsset,
    ExportDiagnostic,
    ExportDocument,
    ExportElement,
    ExportFallback,
    ExportPage,
    ExportProfile,
    ExportProjectMeta,
    ExportProof,
    ExportSource,
)
from app.export.rules import ExportRules, load_export_rules, normalize_export_format
from app.models import BBox, Block, BlockSource, Line, OcrProject, Page
from app.services.export_service import get_export_text, iter_export_blocks, iter_export_lines, iter_export_pages


def build_export_ir(
    project: OcrProject,
    fmt: str = "txt",
    *,
    rules: ExportRules | None = None,
) -> ExportDocument:
    """Build typed Export IR from the internal project model."""
    export_rules = rules or load_export_rules()
    profile = export_rules.profile_for(normalize_export_format(fmt))
    assets: list[ExportAsset] = []
    diagnostics: list[ExportDiagnostic] = []
    pages: list[ExportPage] = []

    for page in iter_export_pages(project):
        elements: list[ExportElement] = []
        if page.error_message:
            diagnostics.append(ExportDiagnostic(
                level="error",
                code="page_error",
                message=page.error_message,
                details={"page_number": page.page_number},
            ))

        for index, block in enumerate(iter_export_blocks(page, include_empty=True), start=1):
            element = _build_element(
                page=page,
                block=block,
                index=index,
                profile=profile,
                rules=export_rules,
                assets=assets,
                diagnostics=diagnostics,
            )
            elements.append(element)

        pages.append(ExportPage(
            page_number=page.page_number,
            source_image=page.display_image_path,
            size={"w": int(page.width), "h": int(page.height)},
            source_meta={
                "source_path": page.source_path,
                "source_type": page.source_type,
                "source_page_index": page.source_page_index,
            },
            status=page.status.value if hasattr(page.status, "value") else str(page.status),
            elements=elements,
        ))

    if not profile.include_diagnostics:
        diagnostics = []
    if not profile.include_assets:
        assets = []

    return ExportDocument(
        version=EXPORT_IR_VERSION,
        profile=profile,
        project=ExportProjectMeta(
            name=project.name,
            page_count=project.page_count,
            summary=project.get_export_summary(),
            created_at=project.created_at,
            updated_at=project.updated_at,
        ),
        pages=pages,
        assets=assets,
        diagnostics=diagnostics,
    )


def _build_element(
    *,
    page: Page,
    block: Block,
    index: int,
    profile: ExportProfile,
    rules: ExportRules,
    assets: list[ExportAsset],
    diagnostics: list[ExportDiagnostic],
) -> ExportElement:
    kind = rules.kind_for_block_type(block.block_type.value)
    rule = rules.rule_for_kind(kind)
    payload_type = str(rule.get("payload") or "text")
    fallback_rule = rules.fallback_for_kind(kind)
    asset_kind = str(rule.get("asset_kind") or "page_region")
    element_id = f"el-p{page.page_number}-{index:04d}"
    lines = list(iter_export_lines(block))
    line_payloads = [_line_payload(line, idx) for idx, line in enumerate(lines)]
    text = "\n".join(item["text"] for item in line_payloads if item["text"].strip())
    fallback: ExportFallback | None = None
    payload: dict[str, Any]

    if payload_type == "text":
        payload = {
            "text": text,
            "lines": line_payloads,
            "inline_role": str(rule.get("inline_role") or "body"),
            "keep_linebreaks": bool(rule.get("keep_linebreaks", kind == "reference")),
        }
    elif payload_type == "asset":
        asset_ref = _append_region_asset(profile, assets, page, block, element_id, asset_kind)
        if kind == "figure":
            payload = {"asset_ref": asset_ref, "alt_text": block.note}
        else:
            payload = {
                "asset_ref": asset_ref,
                "text": text,
                "lines": line_payloads,
            }
            fallback = _fallback_from_rule(fallback_rule, reason="unknown_block_type", asset_ref=asset_ref)
            diagnostics.append(_diagnostic(
                "warning",
                "unknown_block_fallback",
                "未知版面块已保留为图片/区域兜底元素。",
                element_id,
                {"block_type": block.block_type.value},
            ))
        if not asset_ref and kind == "figure":
            fallback = _fallback_from_rule(fallback_rule, reason="missing_asset")
    elif payload_type == "table":
        asset_ref = _append_region_asset(profile, assets, page, block, element_id, asset_kind)
        payload = {
            "mode": str(fallback_rule.get("mode") or "image_fallback"),
            "asset_ref": asset_ref,
            "text": text,
            "lines": line_payloads,
        }
        fallback = _fallback_from_rule(fallback_rule, asset_ref=asset_ref)
        diagnostics.append(_diagnostic(
            "warning",
            "table_fallback_to_image",
            "表格结构化网格不可用，已使用图片兜底导出。",
            element_id,
            {"block_type": block.block_type.value},
        ))
    elif payload_type == "equation":
        if text:
            payload = {
                "mode": "text",
                "text": text,
                "lines": line_payloads,
            }
        else:
            asset_ref = _append_region_asset(profile, assets, page, block, element_id, asset_kind)
            payload = {"mode": str(fallback_rule.get("mode") or "image_fallback"), "asset_ref": asset_ref}
            fallback = _fallback_from_rule(fallback_rule, reason="missing_equation_text", asset_ref=asset_ref)
            diagnostics.append(_diagnostic(
                "warning",
                "equation_fallback_to_image",
                "公式结构化表达不可用，已使用图片兜底导出。",
                element_id,
                {"block_type": block.block_type.value},
            ))
    else:
        asset_ref = _append_region_asset(profile, assets, page, block, element_id, asset_kind)
        payload = {
            "asset_ref": asset_ref,
            "text": text,
            "lines": line_payloads,
        }
        fallback = _fallback_from_rule(fallback_rule, reason="unknown_block_type", asset_ref=asset_ref)
        diagnostics.append(_diagnostic(
            "warning",
            "unknown_block_fallback",
            "未知版面块已保留为图片/区域兜底元素。",
            element_id,
            {"block_type": block.block_type.value},
        ))

    if not lines and payload_type == "text":
        fallback = _fallback_from_strategy(rules, "plain_paragraph", reason="empty_text")
        diagnostics.append(_diagnostic(
            "info",
            "empty_text_element",
            "文本元素没有可导出的行文本。",
            element_id,
            {"block_type": block.block_type.value},
        ))

    if not block.recognizable:
        diagnostics.append(_diagnostic(
            "info",
            "block_not_recognizable",
            "该块标记为不可 OCR，导出仅保留现有内容/兜底。",
            element_id,
            {"block_type": block.block_type.value},
        ))

    return ExportElement(
        id=element_id,
        kind=kind,
        page=page.page_number,
        order=int(block.order),
        bbox=_bbox_to_dict(block.bbox),
        source=_source(page, block, lines),
        proof=_proof(lines),
        payload=payload,
        fallback=fallback,
    )


def _line_payload(line: Line, index: int) -> dict[str, Any]:
    return {
        "line_id": line.id if line.id is not None else index,
        "text": get_export_text(line),
        "ocr_text": line.ocr_text,
        "bbox": _bbox_to_dict(line.bbox),
        "confidence": float(line.confidence),
        "status": line.proof_status.value if hasattr(line.proof_status, "value") else str(line.proof_status),
    }


def _source(page: Page, block: Block, lines: list[Line]) -> ExportSource:
    return ExportSource(
        page_number=page.page_number,
        block_ids=[block.id if block.id is not None else f"p{page.page_number}-block-{block.order}"],
        line_ids=[line.id if line.id is not None else idx for idx, line in enumerate(lines)],
        char_ids=[
            char.id
            for line in lines
            for char in line.chars
            if char.id is not None
        ],
        block_type=block.block_type.value,
        origin=_origin(block.source),
    )


def _origin(source: BlockSource) -> str:
    if source in (BlockSource.MANUAL_DRAW, BlockSource.USER_EDITED):
        return "project_model"
    if source == BlockSource.AUTO_TIGHTENED:
        return "derived"
    return "project_model"


def _proof(lines: list[Line]) -> ExportProof:
    if not lines:
        return ExportProof()
    confidence = sum(line.confidence for line in lines) / len(lines)
    flags = sorted({flag for line in lines for flag in line.review_flags})
    corrected = any(bool(line.original_text) and line.original_text != get_export_text(line) for line in lines)
    status_order = ["auto_flagged", "unchecked", "modified", "ok"]
    statuses = [line.proof_status.value if hasattr(line.proof_status, "value") else str(line.proof_status) for line in lines]
    status = min(statuses, key=lambda item: status_order.index(item) if item in status_order else 99)
    return ExportProof(status=status, confidence=confidence, flags=flags, corrected=corrected)


def _append_region_asset(
    profile: ExportProfile,
    assets: list[ExportAsset],
    page: Page,
    block: Block,
    element_id: str,
    asset_kind: str,
) -> str | None:
    if not profile.include_assets:
        return None
    asset_id = f"asset-{element_id}"
    assets.append(ExportAsset(
        id=asset_id,
        kind=asset_kind,
        page_number=page.page_number,
        bbox=_bbox_to_dict(block.bbox),
        path=page.display_image_path,
        mime=_mime_for_path(page.display_image_path),
    ))
    return asset_id


def _mime_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".tif" or suffix == ".tiff":
        return "image/tiff"
    return "image/png"


def _bbox_to_dict(bbox: BBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    return {"x": int(bbox.x), "y": int(bbox.y), "w": int(bbox.w), "h": int(bbox.h)}


def _fallback(reason: str, mode: str, asset_ref: str | None = None) -> ExportFallback:
    return ExportFallback(used=True, reason=reason, mode=mode, asset_ref=asset_ref)


def _fallback_from_rule(
    rule: dict[str, Any],
    *,
    reason: str | None = None,
    asset_ref: str | None = None,
) -> ExportFallback:
    return _fallback(
        reason or str(rule.get("reason") or "fallback"),
        str(rule.get("mode") or "fallback"),
        asset_ref,
    )


def _fallback_from_strategy(
    rules: ExportRules,
    name: str,
    *,
    reason: str | None = None,
    asset_ref: str | None = None,
) -> ExportFallback:
    return _fallback_from_rule(rules.fallback_strategies.get(name, {}), reason=reason, asset_ref=asset_ref)


def _diagnostic(
    level: str,
    code: str,
    message: str,
    element_id: str,
    details: dict[str, Any],
) -> ExportDiagnostic:
    return ExportDiagnostic(level=level, code=code, message=message, element_id=element_id, details=details)
