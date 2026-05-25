from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines
from app.models import OcrProject


class TxtExporter(ExporterBase):
    """纯文本导出：只输出最终文本，不混入调试元数据。"""

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "txt")
        lines: list[str] = []
        for page in document.pages:
            for element in page.elements:
                texts = element_lines(element)
                if not texts:
                    continue
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend(texts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + ("\n" if lines else ""))
