"""XML 导出：使用 lxml，保留完整 bbox 坐标。"""
from lxml import etree

from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines
from app.models import OcrProject


class XmlExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "xml")
        root = etree.Element("OcrDocument")
        root.set("version", document.version)
        root.set("format", document.profile.format)
        root.set("mode", document.profile.mode)
        root.set("name", document.project.name)
        root.set("pages", str(document.project.page_count))

        for page in document.pages:
            page_el = etree.SubElement(root, "Page")
            page_el.set("number", str(page.page_number))
            page_el.set("image", page.source_image)
            page_el.set("width", str(page.size["w"]))
            page_el.set("height", str(page.size["h"]))

            for element in page.elements:
                block_el = etree.SubElement(page_el, "Element")
                block_el.set("id", element.id)
                block_el.set("kind", element.kind)
                block_el.set("order", str(element.order))
                if element.bbox:
                    block_el.set("x", str(element.bbox["x"]))
                    block_el.set("y", str(element.bbox["y"]))
                    block_el.set("w", str(element.bbox["w"]))
                    block_el.set("h", str(element.bbox["h"]))
                if element.proof:
                    block_el.set("confidence", f"{(element.proof.confidence or 0):.4f}")
                    block_el.set("status", element.proof.status)
                    block_el.set("corrected", "true" if element.proof.corrected else "false")
                if element.fallback and element.fallback.used:
                    fallback_el = etree.SubElement(block_el, "Fallback")
                    fallback_el.set("reason", element.fallback.reason)
                    fallback_el.set("mode", element.fallback.mode)
                    if element.fallback.asset_ref:
                        fallback_el.set("asset_ref", element.fallback.asset_ref)

                payload_el = etree.SubElement(block_el, "Payload")
                payload_el.set("mode", str(element.payload.get("mode") or "text"))
                payload_lines = element.payload.get("lines") or []
                if payload_lines:
                    line_items = [(str(item.get("text") or ""), item) for item in payload_lines]
                else:
                    line_items = [(text, {}) for text in element_lines(element)]
                for text, item in line_items:
                    line_el = etree.SubElement(block_el, "Line")
                    bbox = item.get("bbox") if isinstance(item, dict) else None
                    if bbox:
                        line_el.set("x", str(bbox.get("x", 0)))
                        line_el.set("y", str(bbox.get("y", 0)))
                        line_el.set("w", str(bbox.get("w", 0)))
                        line_el.set("h", str(bbox.get("h", 0)))
                    if isinstance(item, dict):
                        line_el.set("confidence", f"{float(item.get('confidence') or 0):.4f}")
                        line_el.set("status", str(item.get("status") or ""))
                    line_el.text = text

        assets_el = etree.SubElement(root, "Assets")
        for asset in document.assets:
            asset_el = etree.SubElement(assets_el, "Asset")
            asset_el.set("id", asset.id)
            asset_el.set("kind", asset.kind)
            asset_el.set("page_number", str(asset.page_number))
            asset_el.set("path", asset.path)
            if asset.bbox:
                asset_el.set("bbox", f"{asset.bbox['x']},{asset.bbox['y']},{asset.bbox['w']},{asset.bbox['h']}")

        diagnostics_el = etree.SubElement(root, "Diagnostics")
        for diagnostic in document.diagnostics:
            diagnostic_el = etree.SubElement(diagnostics_el, "Diagnostic")
            diagnostic_el.set("level", diagnostic.level)
            diagnostic_el.set("code", diagnostic.code)
            if diagnostic.element_id:
                diagnostic_el.set("element_id", diagnostic.element_id)
            diagnostic_el.text = diagnostic.message

        tree = etree.ElementTree(root)
        tree.write(
            out_path,
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )
