"""Build the pure Export IR from one immutable export snapshot."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

from app.core.paddle_labels import normalize_paddle_label
from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
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
from app.models.export_snapshot import ExportPageSnapshot, ExportProjectSnapshot
from app.models.geometry import BBox
from app.models.layout_snapshot import LayoutBlockSnapshot
from app.models.ocr_records import OcrAtom, OcrCandidate, OcrLine
from app.models.project_session import TableTextRecord
from app.services.export_service import (
    build_export_summary,
    export_line_facts,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)


_BLOCK_TYPE_TO_KIND = {
    "text": "paragraph",
    "title": "title",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "table": "table",
    "table_caption": "table_caption",
    "reference": "reference",
    "equation": "equation",
    "unknown": "unknown",
}

_LABEL_TO_BLOCK_TYPE = {
    "text": "text",
    "paragraph_text": "text",
    "text_block": "text",
    "text_box": "text",
    "body": "text",
    "paragraph": "text",
    "plain_text": "text",
    "doc_text": "text",
    "text_region": "text",
    "body_text": "text",
    "content": "text",
    "abstract": "text",
    "table_of_contents": "text",
    "toc": "text",
    "page_number": "text",
    "number": "text",
    "header": "text",
    "footer": "text",
    "footnote": "text",
    "vision_footnote": "text",
    "sidebar_text": "text",
    "vertical_text": "text",
    "algorithm": "text",
    "title": "title",
    "doc_title": "title",
    "doc_heading": "title",
    "section_title": "title",
    "chapter_title": "title",
    "paragraph_title": "title",
    "text_title": "title",
    "heading": "title",
    "headline": "title",
    "heading_1": "title",
    "heading_2": "title",
    "heading_3": "title",
    "heading_4": "title",
    "heading_5": "title",
    "heading_6": "title",
    "figure": "figure",
    "graphic": "figure",
    "photo": "figure",
    "logo": "figure",
    "image": "figure",
    "picture": "figure",
    "illustration": "figure",
    "chart": "figure",
    "seal": "figure",
    "header_image": "figure",
    "footer_image": "figure",
    "figure_caption": "figure_caption",
    "figure_title": "figure_caption",
    "caption": "figure_caption",
    "figure_note": "figure_caption",
    "image_caption": "figure_caption",
    "table": "table",
    "table_region": "table",
    "table_block": "table",
    "table_cell": "table",
    "table_body": "table",
    "table_caption": "table_caption",
    "table_caption_text": "table_caption",
    "table_note": "table_caption",
    "table_title": "table_caption",
    "reference": "reference",
    "reference_content": "reference",
    "references": "reference",
    "reference_list": "reference",
    "reference_text": "reference",
    "bibliography": "reference",
    "equation": "equation",
    "equation_block": "equation",
    "display_formula": "equation",
    "isolated_formula": "equation",
    "inline_formula": "equation",
    "formula": "equation",
    "formula_block": "equation",
    "formula_number": "equation",
    "math": "equation",
    "math_formula": "equation",
    "math_block": "equation",
}


def build_export_ir(
    snapshot: ExportProjectSnapshot,
    fmt: str = "txt",
    *,
    rules: ExportRules | None = None,
) -> ExportDocument:
    """Build typed Export IR from the supplied snapshot and nothing else."""
    if not isinstance(snapshot, ExportProjectSnapshot):
        raise TypeError("export requires ExportProjectSnapshot")
    export_rules = rules or load_export_rules()
    normalized_format = normalize_export_format(fmt)
    profile = export_rules.profile_for(normalized_format)
    private_paths = normalized_format in {"json", "xml", "html"}
    assets: list[ExportAsset] = []
    diagnostics: list[ExportDiagnostic] = []
    pages: list[ExportPage] = []

    for page_snapshot in iter_export_pages(snapshot):
        page = page_snapshot.page
        elements: list[ExportElement] = []
        if page.error:
            diagnostics.append(ExportDiagnostic(
                level="error",
                code="page_error",
                message=page.error,
                details={"page_number": page.page_number},
            ))

        for index, block in enumerate(
            iter_export_blocks(page_snapshot, include_empty=True),
            start=1,
        ):
            elements.append(_build_element(
                page_snapshot=page_snapshot,
                block=block,
                index=index,
                profile=profile,
                rules=export_rules,
                assets=assets,
                diagnostics=diagnostics,
                private_paths=private_paths,
            ))

        pages.append(ExportPage(
            page_number=page.page_number,
            source_image=_export_path(page.image_path, private_paths=private_paths),
            size={"w": int(page.width), "h": int(page.height)},
            source_meta={
                "page_uid": page.uid,
                "source_path": _export_path(page.source_path, private_paths=private_paths),
                "source_page_index": page.source_page_index,
                "image_hash": page.image_hash,
                "image_revision": page.image_revision,
            },
            status=page.status,
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
            name=snapshot.project.name,
            page_count=len(snapshot.pages),
            summary=build_export_summary(snapshot),
        ),
        pages=pages,
        assets=assets,
        diagnostics=diagnostics,
    )


def _build_element(
    *,
    page_snapshot: ExportPageSnapshot,
    block: LayoutBlockSnapshot,
    index: int,
    profile: ExportProfile,
    rules: ExportRules,
    assets: list[ExportAsset],
    diagnostics: list[ExportDiagnostic],
    private_paths: bool,
) -> ExportElement:
    attrs = _layout_attrs(block)
    kind = rules.kind_for_block_type(attrs["semantic_block_type"])
    rule = rules.rule_for_kind(kind)
    payload_type = str(rule.get("payload") or "text")
    fallback_rule = rules.fallback_for_kind(kind)
    asset_kind = str(rule.get("asset_kind") or "page_region")
    page = page_snapshot.page
    element_id = f"el-p{page.page_number}-{index:04d}"
    lines = tuple(iter_export_lines(page_snapshot, block))
    line_payloads = [
        _line_payload(page_snapshot, line, line_index)
        for line_index, line in enumerate(lines)
    ]
    table_records = _table_text_records(page_snapshot, block)
    text = _element_text(line_payloads, table_records)
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
        asset_ref = _append_region_asset(
            profile,
            assets,
            page,
            block,
            element_id,
            asset_kind,
            private_paths=private_paths,
        )
        if kind == "figure":
            payload = {
                "asset_ref": asset_ref,
                "alt_text": _figure_alt_text(page_snapshot, block),
            }
        else:
            payload = {"asset_ref": asset_ref, "text": text, "lines": line_payloads}
            fallback = _fallback_from_rule(fallback_rule, reason="unknown_block_type", asset_ref=asset_ref)
            diagnostics.append(_diagnostic(
                "warning",
                "unknown_block_fallback",
                "未知版面块已保留为图片/区域兜底元素。",
                element_id,
                {"block_attributes": attrs},
            ))
        if not asset_ref and kind == "figure":
            fallback = _fallback_from_rule(fallback_rule, reason="missing_asset")
    elif payload_type == "table":
        asset_ref = _append_region_asset(
            profile,
            assets,
            page,
            block,
            element_id,
            asset_kind,
            private_paths=private_paths,
        )
        payload = {
            "mode": str(fallback_rule.get("mode") or "image_fallback"),
            "asset_ref": asset_ref,
            "text": text,
            "lines": line_payloads,
        }
        table_cells = _table_text_layer_cells(table_records)
        if table_cells:
            payload[TABLE_TEXT_LAYER_CELLS_KEY] = table_cells
        fallback = _fallback_from_rule(fallback_rule, asset_ref=asset_ref)
        diagnostics.append(_diagnostic(
            "warning",
            "table_fallback_to_image",
            "表格结构化网格不可用，已使用图片兜底导出。",
            element_id,
            {"block_attributes": attrs},
        ))
    elif payload_type == "equation":
        if text:
            payload = {"mode": "text", "text": text, "lines": line_payloads}
        else:
            asset_ref = _append_region_asset(
                profile,
                assets,
                page,
                block,
                element_id,
                asset_kind,
                private_paths=private_paths,
            )
            payload = {
                "mode": str(fallback_rule.get("mode") or "image_fallback"),
                "asset_ref": asset_ref,
            }
            fallback = _fallback_from_rule(
                fallback_rule,
                reason="missing_equation_text",
                asset_ref=asset_ref,
            )
            diagnostics.append(_diagnostic(
                "warning",
                "equation_fallback_to_image",
                "公式结构化表达不可用，已使用图片兜底导出。",
                element_id,
                {"block_attributes": attrs},
            ))
    else:
        asset_ref = _append_region_asset(
            profile,
            assets,
            page,
            block,
            element_id,
            asset_kind,
            private_paths=private_paths,
        )
        payload = {"asset_ref": asset_ref, "text": text, "lines": line_payloads}
        fallback = _fallback_from_rule(fallback_rule, reason="unknown_block_type", asset_ref=asset_ref)
        diagnostics.append(_diagnostic(
            "warning",
            "unknown_block_fallback",
            "未知版面块已保留为图片/区域兜底元素。",
            element_id,
            {"block_attributes": attrs},
        ))

    if not lines and payload_type == "text":
        fallback = _fallback_from_strategy(rules, "plain_paragraph", reason="empty_text")
        diagnostics.append(_diagnostic(
            "info",
            "empty_text_element",
            "文本元素没有可导出的行文本。",
            element_id,
            {"block_attributes": attrs},
        ))

    if _enum_value(block.ocr_policy) != "text_ocr":
        diagnostics.append(_diagnostic(
            "info",
            "block_not_text_ocr_policy",
            "该块 OCR 策略不是 text_ocr，导出仅保留现有内容/兜底。",
            element_id,
            {"block_attributes": attrs, "ocr_policy": _enum_value(block.ocr_policy)},
        ))

    return ExportElement(
        id=element_id,
        kind=kind,
        page=page.page_number,
        order=int(block.order),
        bbox=_xyxy_to_dict(block.bbox.to_xyxy()),
        source=_source(page_snapshot, block, lines, attrs),
        proof=_proof(page_snapshot, lines),
        payload=payload,
        layout_attributes=attrs,
        fallback=fallback,
    )


def _line_payload(
    page_snapshot: ExportPageSnapshot,
    line: OcrLine,
    line_index: int,
) -> dict[str, Any]:
    facts = export_line_facts(page_snapshot, line)
    return {
        "line_id": line.uid or line_index,
        "text": get_export_text(page_snapshot, line),
        "ocr_text": line.text,
        "bbox": _xyxy_to_dict(line.bbox),
        "chars": _char_payloads(page_snapshot, line),
        "confidence": float(line.confidence),
        "status": facts.status,
    }


def _char_payloads(page_snapshot: ExportPageSnapshot, line: OcrLine) -> list[dict[str, Any]]:
    atom_by_uid = {atom.uid: atom for atom in page_snapshot.ocr_atoms if atom.line_uid == line.uid}
    candidate_by_atom: dict[str, list[OcrCandidate]] = {}
    for candidate in page_snapshot.ocr_candidates:
        if candidate.atom_uid in atom_by_uid:
            candidate_by_atom.setdefault(candidate.atom_uid, []).append(candidate)

    atoms = _atoms_for_line(line, atom_by_uid)
    return [_char_payload(atom, candidate_by_atom.get(atom.uid, [])) for atom in atoms]


def _atoms_for_line(
    line: OcrLine,
    atom_by_uid: dict[str, OcrAtom],
) -> list[OcrAtom]:
    declared_uids = set(line.atom_uids)
    atoms = [atom_by_uid[uid] for uid in line.atom_uids if uid in atom_by_uid]
    atoms.extend(
        sorted(
            (atom for uid, atom in atom_by_uid.items() if uid not in declared_uids),
            key=lambda atom: (atom.index, atom.uid),
        )
    )
    return atoms


def _char_payload(atom: OcrAtom, candidates: list[OcrCandidate]) -> dict[str, Any]:
    return {
        "char_id": atom.uid,
        "char": atom.text,
        "bbox": _xyxy_to_dict(atom.bbox),
        "confidence": float(atom.confidence),
        "bbox_source": "ocr_atom",
        "bbox_granularity": "char",
        "token_text": atom.text,
        "candidates": [
            {
                "candidate_id": candidate.uid,
                "text": candidate.text,
                "confidence": float(candidate.confidence),
                "rank": candidate.rank,
                "source": candidate.source,
                "bbox": _xyxy_to_dict(candidate.bbox),
            }
            for candidate in sorted(candidates, key=lambda item: (item.rank, item.uid))
        ],
    }


def _element_text(
    line_payloads: list[dict[str, Any]],
    table_records: tuple[TableTextRecord, ...],
) -> str:
    line_text = "\n".join(
        str(item["text"])
        for item in line_payloads
        if str(item["text"]).strip()
    )
    if line_text:
        return line_text
    return "\n".join(record.text for record in table_records if record.text.strip())


def _table_text_records(
    page_snapshot: ExportPageSnapshot,
    block: LayoutBlockSnapshot,
) -> tuple[TableTextRecord, ...]:
    return tuple(sorted(
        (
            record
            for record in page_snapshot.table_texts
            if record.page_uid == page_snapshot.page.uid and record.table_uid == block.uid
        ),
        key=lambda record: (record.row_index, record.column_index, record.uid),
    ))


def _table_text_layer_cells(records: tuple[TableTextRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "text": record.text,
            "row": record.row_index,
            "col": record.column_index,
            "row_span": 1,
            "col_span": 1,
            "bbox_source": "table_text_record",
            "bbox_granularity": "table_cell",
        }
        for record in records
        if record.text
    ]


def _figure_alt_text(page_snapshot: ExportPageSnapshot, block: LayoutBlockSnapshot) -> str:
    regions = {
        region.uid: region
        for region in page_snapshot.ocr_regions
    }
    for binding in page_snapshot.bindings:
        if binding.source_uid != block.uid or binding.relation != "observed_by":
            continue
        region = regions.get(binding.target_uid)
        if region is not None:
            return region.text_hint or region.label
    return ""


def _source(
    page_snapshot: ExportPageSnapshot,
    block: LayoutBlockSnapshot,
    lines: tuple[OcrLine, ...],
    attrs: dict[str, Any],
) -> ExportSource:
    return ExportSource(
        page_number=page_snapshot.page.page_number,
        block_ids=[block.uid],
        line_ids=[line.uid for line in lines],
        char_ids=[
            atom.uid
            for line in lines
            for atom in _atoms_for_line(
                line,
                {
                    atom.uid: atom
                    for atom in page_snapshot.ocr_atoms
                    if atom.line_uid == line.uid
                },
            )
        ],
        block_type=_enum_value(block.block_type),
        source_label=attrs["source_label"],
        semantic_label=attrs["semantic_label"],
        semantic_block_type=attrs["semantic_block_type"],
        origin=str(block.origin.created_by or ""),
    )


def _layout_attrs(block: LayoutBlockSnapshot) -> dict[str, Any]:
    block_type = _enum_value(block.block_type)
    source_label = normalize_paddle_label(block.source_label)
    semantic_label = source_label or normalize_paddle_label(block_type)
    semantic_block_type = _LABEL_TO_BLOCK_TYPE.get(semantic_label, block_type)
    if semantic_block_type not in _BLOCK_TYPE_TO_KIND:
        semantic_block_type = "unknown"
    return {
        "block_type": block_type,
        "source_label": source_label,
        "current_label": normalize_paddle_label(block.source_label or block_type),
        "origin_label": normalize_paddle_label(block.origin.vendor_label),
        "raw_label": "",
        "semantic_label": semantic_label,
        "semantic_block_type": semantic_block_type,
    }


def _proof(page_snapshot: ExportPageSnapshot, lines: tuple[OcrLine, ...]) -> ExportProof:
    if not lines:
        return ExportProof()
    facts = [export_line_facts(page_snapshot, line) for line in lines]
    status_order = {"auto_flagged": 0, "unchecked": 1, "modified": 2, "ok": 3}
    status = min(
        (fact.status for fact in facts),
        key=lambda value: status_order.get(value, 99),
    )
    return ExportProof(
        status=status,
        confidence=sum(line.confidence for line in lines) / len(lines),
        flags=sorted({flag for fact in facts for flag in fact.flags}),
        corrected=any(fact.corrected for fact in facts),
    )


def _append_region_asset(
    profile: ExportProfile,
    assets: list[ExportAsset],
    page,
    block: LayoutBlockSnapshot,
    element_id: str,
    asset_kind: str,
    *,
    private_paths: bool,
) -> str | None:
    if not profile.include_assets:
        return None
    asset_id = f"asset-{element_id}"
    assets.append(ExportAsset(
        id=asset_id,
        kind=asset_kind,
        page_number=page.page_number,
        bbox=_xyxy_to_dict(block.bbox.to_xyxy()),
        path=_export_path(page.image_path, private_paths=private_paths),
        mime=_mime_for_path(page.image_path),
    ))
    return asset_id


def _export_path(path: str, *, private_paths: bool) -> str:
    if not private_paths or not path:
        return path
    raw = str(path)
    posix_name = Path(raw).name
    windows_name = PureWindowsPath(raw).name
    if "\\" in raw or ":" in raw:
        return windows_name or posix_name
    return posix_name


def _mime_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "image/png"


def _xyxy_to_dict(bbox: tuple[int, int, int, int] | BBox | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    if isinstance(bbox, BBox):
        return bbox.to_dict()
    x1, y1, x2, y2 = bbox
    return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


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
    return _fallback_from_rule(
        rules.fallback_strategies.get(name, {}),
        reason=reason,
        asset_ref=asset_ref,
    )


def _diagnostic(
    level: str,
    code: str,
    message: str,
    element_id: str,
    details: dict[str, Any],
) -> ExportDiagnostic:
    return ExportDiagnostic(
        level=level,
        code=code,
        message=message,
        element_id=element_id,
        details=details,
    )


__all__ = ["build_export_ir"]
