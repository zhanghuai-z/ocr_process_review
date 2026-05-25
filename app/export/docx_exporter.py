"""DOCX 导出：python-docx，按块段落输出。"""
from docx import Document
from docx.shared import Pt, RGBColor

from app.export.base import ExporterBase
from app.export.ir import ExportElement
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines
from app.models import OcrProject


class DocxExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "docx")
        doc = Document()
        doc.core_properties.title = project.name
        doc.core_properties.author = "OCR Process"

        for page in document.pages:
            doc.add_heading(f"第 {page.page_number} 页", level=1)

            for element in page.elements:
                self._add_element(doc, element)
                doc.add_paragraph()

            doc.add_page_break()

        doc.save(out_path)

    def _add_element(self, doc: Document, element: ExportElement) -> None:
        texts = element_lines(element)
        if element.kind == "title":
            for text in texts:
                doc.add_heading(text, level=2)
            return

        para = _safe_add_paragraph(doc, _docx_style(element.kind))
        for text in texts:
            run = para.add_run(text + "\n")
            run.font.size = Pt(_font_size(element.kind))
            run.italic = element.kind in {"figure_caption", "table_caption"}
            if element.proof and element.proof.status == "auto_flagged":
                run.font.color.rgb = RGBColor(0xF4, 0x43, 0x36)
            elif element.proof and element.proof.status == "modified":
                run.font.color.rgb = RGBColor(0xFF, 0xA7, 0x26)


def _safe_add_paragraph(doc: Document, style_name: str):
    try:
        return doc.add_paragraph(style=style_name)
    except KeyError:
        return doc.add_paragraph()


def _docx_style(kind: str) -> str:
    if kind in {"figure_caption", "table_caption"}:
        return "Caption"
    return "Normal"


def _font_size(kind: str) -> int:
    if kind in {"figure_caption", "table_caption"}:
        return 10
    if kind == "reference":
        return 11
    return 12
