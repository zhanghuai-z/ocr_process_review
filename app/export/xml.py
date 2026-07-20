"""XML primary archive renderer."""
from __future__ import annotations

import json
from typing import Any

from lxml import etree

from app.export.base import ExporterBase
from app.export.ir import ExportDocument
from app.export.ir_builder import build_export_ir
from app.models.export_snapshot import ExportProjectSnapshot


class XmlExporter(ExporterBase):
    """Write the XML authority archive for Export IR semantics."""

    def export(self, snapshot: ExportProjectSnapshot, out_path: str) -> None:
        document = build_export_ir(snapshot, "xml")
        root = _build_xml_archive(document)
        tree = etree.ElementTree(root)
        tree.write(
            out_path,
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )


def _build_xml_archive(document: ExportDocument):
    root = etree.Element("OcrArchive")
    root.set("version", document.version)
    root.set("format", document.profile.format)
    root.set("mode", document.profile.mode)
    root.set("archive_role", document.profile.archive_role)
    root.set("authority", document.profile.authority)
    root.set("parity_group", document.profile.parity_group)

    project_el = etree.SubElement(root, "Project")
    project_el.set("name", document.project.name)
    project_el.set("page_count", str(document.project.page_count))
    if document.project.created_at is not None:
        project_el.set("created_at", str(document.project.created_at))
    if document.project.updated_at is not None:
        project_el.set("updated_at", str(document.project.updated_at))
    _append_mapping(etree.SubElement(project_el, "Summary"), document.project.summary)

    pages_el = etree.SubElement(root, "Pages")
    for page in document.pages:
        page_el = etree.SubElement(pages_el, "Page")
        page_el.set("number", str(page.page_number))
        page_el.set("source_image", page.source_image)
        page_el.set("width", str(page.size["w"]))
        page_el.set("height", str(page.size["h"]))
        page_el.set("status", page.status)
        _append_mapping(etree.SubElement(page_el, "SourceMeta"), page.source_meta)

        elements_el = etree.SubElement(page_el, "Elements")
        for element in page.elements:
            element_el = etree.SubElement(elements_el, "Element")
            element_el.set("id", element.id)
            element_el.set("kind", element.kind)
            element_el.set("page", str(element.page))
            element_el.set("order", str(element.order))
            _append_bbox(element_el, "BBox", element.bbox)

            source_el = etree.SubElement(element_el, "Source")
            source_el.set("page_number", str(element.source.page_number))
            source_el.set("origin", element.source.origin)
            source_el.set("block_type", element.source.block_type)
            source_el.set("source_label", element.source.source_label)
            source_el.set("semantic_label", element.source.semantic_label)
            source_el.set("semantic_block_type", element.source.semantic_block_type)
            _append_id_list(source_el, "BlockIds", "BlockId", element.source.block_ids)
            _append_id_list(source_el, "LineIds", "LineId", element.source.line_ids)
            _append_id_list(source_el, "CharIds", "CharId", element.source.char_ids)

            if element.proof:
                proof_el = etree.SubElement(element_el, "Proof")
                proof_el.set("status", element.proof.status)
                proof_el.set("corrected", "true" if element.proof.corrected else "false")
                proof_el.set("confidence", str(element.proof.confidence))
                flags_el = etree.SubElement(proof_el, "Flags")
                for flag in element.proof.flags:
                    etree.SubElement(flags_el, "Flag").text = flag

            payload_el = etree.SubElement(element_el, "Payload")
            _append_payload(payload_el, element.payload)

            if element.fallback and element.fallback.used:
                fallback_el = etree.SubElement(element_el, "Fallback")
                fallback_el.set("used", "true")
                fallback_el.set("reason", element.fallback.reason)
                fallback_el.set("mode", element.fallback.mode)
                if element.fallback.asset_ref:
                    fallback_el.set("asset_ref", element.fallback.asset_ref)

    assets_el = etree.SubElement(root, "Assets")
    for asset in document.assets:
        asset_el = etree.SubElement(assets_el, "Asset")
        asset_el.set("id", asset.id)
        asset_el.set("kind", asset.kind)
        asset_el.set("page_number", str(asset.page_number))
        asset_el.set("path", asset.path)
        asset_el.set("mime", asset.mime)
        _append_bbox(asset_el, "BBox", asset.bbox)

    diagnostics_el = etree.SubElement(root, "Diagnostics")
    for diagnostic in document.diagnostics:
        diagnostic_el = etree.SubElement(diagnostics_el, "Diagnostic")
        diagnostic_el.set("level", diagnostic.level)
        diagnostic_el.set("code", diagnostic.code)
        if diagnostic.element_id:
            diagnostic_el.set("element_id", diagnostic.element_id)
        diagnostic_el.text = diagnostic.message
        _append_mapping(etree.SubElement(diagnostic_el, "Details"), diagnostic.details)

    return root


def _append_payload(parent, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key == "lines" and isinstance(value, list):
            lines_el = etree.SubElement(parent, "Lines")
            for item in value:
                line_el = etree.SubElement(lines_el, "Line")
                line_el.set("line_id", str(item.get("line_id", "")))
                line_el.set("confidence", str(item.get("confidence", "")))
                line_el.set("status", str(item.get("status", "")))
                _append_bbox(line_el, "BBox", item.get("bbox"))
                if "ocr_text" in item:
                    etree.SubElement(line_el, "OcrText").text = str(item.get("ocr_text") or "")
                line_el.text = str(item.get("text") or "")
            continue
        _append_field(parent, key, value)


def _append_mapping(parent, data: dict[str, Any]) -> None:
    for key, value in data.items():
        _append_field(parent, key, value)


def _append_field(parent, key: str, value: Any) -> None:
    field_el = etree.SubElement(parent, "Field")
    field_el.set("name", key)
    field_el.set("type", type(value).__name__)
    if isinstance(value, (dict, list)):
        field_el.text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        field_el.text = "" if value is None else str(value)


def _append_bbox(parent, tag: str, bbox: dict[str, int] | None) -> None:
    if not bbox:
        return
    bbox_el = etree.SubElement(parent, tag)
    bbox_el.set("x", str(bbox.get("x", 0)))
    bbox_el.set("y", str(bbox.get("y", 0)))
    bbox_el.set("w", str(bbox.get("w", 0)))
    bbox_el.set("h", str(bbox.get("h", 0)))


def _append_id_list(parent, container_tag: str, item_tag: str, values: list[int | str]) -> None:
    container_el = etree.SubElement(parent, container_tag)
    for value in values:
        item_el = etree.SubElement(container_el, item_tag)
        item_el.text = str(value)
