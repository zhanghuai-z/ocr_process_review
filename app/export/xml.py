"""XML 导出：使用 lxml，保留完整 bbox 坐标。"""
from lxml import etree

from app.export.base import ExporterBase
from app.models import OcrProject
from app.services.export_service import (
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)


class XmlExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        root = etree.Element("OcrDocument")
        root.set("name", project.name)
        root.set("pages", str(project.page_count))

        for page in iter_export_pages(project):
            page_el = etree.SubElement(root, "Page")
            page_el.set("number", str(page.page_number))
            page_el.set("image", page.image_path)
            page_el.set("width", str(page.width))
            page_el.set("height", str(page.height))

            for block in iter_export_blocks(page, include_empty=True):
                block_el = etree.SubElement(page_el, "Block")
                block_el.set("type", block.block_type.value)
                bb = block.bbox
                block_el.set("x", str(bb.x))
                block_el.set("y", str(bb.y))
                block_el.set("w", str(bb.w))
                block_el.set("h", str(bb.h))
                block_el.set("order", str(block.order))
                block_el.set("confidence", f"{block.avg_confidence:.4f}")

                for line in iter_export_lines(block):
                    line_el = etree.SubElement(block_el, "Line")
                    bb2 = line.bbox
                    line_el.set("x", str(bb2.x))
                    line_el.set("y", str(bb2.y))
                    line_el.set("w", str(bb2.w))
                    line_el.set("h", str(bb2.h))
                    line_el.set("confidence", f"{line.confidence:.4f}")
                    line_el.set("status", line.proof_status.value)
                    line_el.text = get_export_text(line)

                    for char in line.chars:
                        char_el = etree.SubElement(line_el, "Char")
                        char_el.set("confidence", f"{char.confidence:.4f}")
                        char_el.text = char.char

        tree = etree.ElementTree(root)
        tree.write(
            out_path,
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )
