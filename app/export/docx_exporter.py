"""DOCX 导出：python-docx，按块段落输出。"""
from docx import Document
from docx.shared import Pt, RGBColor

from app.export.base import ExporterBase
from app.models import OcrProject, ProofStatus


class DocxExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        doc = Document()
        doc.core_properties.title = project.name
        doc.core_properties.author = "OCR Process"

        for page in project.pages:
            heading = doc.add_heading(f"第 {page.page_number} 页", level=1)

            for block in page.text_blocks:
                para = doc.add_paragraph()
                for line in block.lines:
                    run = para.add_run(line.text + "\n")
                    run.font.size = Pt(12)
                    # 标注颜色
                    if line.proof_status == ProofStatus.AUTO_FLAGGED:
                        run.font.color.rgb = RGBColor(0xF4, 0x43, 0x36)
                    elif line.proof_status == ProofStatus.MODIFIED:
                        run.font.color.rgb = RGBColor(0xFF, 0xA7, 0x26)

                doc.add_paragraph()  # 块间空行

            doc.add_page_break()

        doc.save(out_path)
