"""DOCX 导出：python-docx，按块段落输出。"""
from docx import Document
from docx.shared import Pt, RGBColor

from app.export.base import ExporterBase
from app.models import Block, OcrProject, ProofStatus
from app.services.export_service import (
    get_block_style,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)


class DocxExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        doc = Document()
        doc.core_properties.title = project.name
        doc.core_properties.author = "OCR Process"

        for page in iter_export_pages(project):
            doc.add_heading(f"第 {page.page_number} 页", level=1)

            for block in iter_export_blocks(page):
                self._add_block(doc, block)
                doc.add_paragraph()

            doc.add_page_break()

        doc.save(out_path)

    def _add_block(self, doc: Document, block: Block) -> None:
        style = get_block_style(block)
        texts = [get_export_text(line) for line in iter_export_lines(block)]
        if block.block_type.value == "title":
            for text in texts:
                doc.add_heading(text, level=2)
            return

        para = _safe_add_paragraph(doc, style.docx_style)
        for line in iter_export_lines(block):
            run = para.add_run(get_export_text(line) + "\n")
            run.font.size = Pt(style.font_size_pt)
            run.italic = style.italic
            if line.proof_status == ProofStatus.AUTO_FLAGGED:
                run.font.color.rgb = RGBColor(0xF4, 0x43, 0x36)
            elif line.proof_status == ProofStatus.MODIFIED:
                run.font.color.rgb = RGBColor(0xFF, 0xA7, 0x26)


def _safe_add_paragraph(doc: Document, style_name: str):
    try:
        return doc.add_paragraph(style=style_name)
    except KeyError:
        return doc.add_paragraph()
