from app.export.base import ExporterBase
from app.models import OcrProject


class TxtExporter(ExporterBase):
    """纯文本导出：按页、按块、按行输出。"""

    def export(self, project: OcrProject, out_path: str) -> None:
        lines = []
        for page in project.pages:
            lines.append(f"=== 第 {page.page_number} 页 ===")
            for block in page.text_blocks:
                for line in block.lines:
                    lines.append(line.text)
                lines.append("")  # 块间空行
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
