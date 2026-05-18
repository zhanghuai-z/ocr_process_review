from app.export.base import ExporterBase
from app.models import OcrProject
from app.services.export_service import (
    format_bbox,
    get_block_label,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)


class TxtExporter(ExporterBase):
    """纯文本导出：按页、按块、按行输出。"""

    def export(self, project: OcrProject, out_path: str) -> None:
        lines = []
        for page in iter_export_pages(project):
            lines.append(f"=== 第 {page.page_number} 页 ===")
            for block in iter_export_blocks(page):
                lines.append(
                    f"[{get_block_label(block)} #{block.order} bbox={format_bbox(block.bbox)}]"
                )
                for line in iter_export_lines(block):
                    lines.append(get_export_text(line))
                lines.append("")  # 块间空行
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
