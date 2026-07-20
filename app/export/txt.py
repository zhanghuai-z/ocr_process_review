from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.ir import ExportElement
from app.export.rendering import element_lines, join_reflow_text_lines
from app.models.export_snapshot import ExportProjectSnapshot


def txt_output_paths(out_path: str) -> tuple[str, str]:
    if out_path.endswith(".utf8.txt"):
        stem = out_path[:-len(".utf8.txt")]
    elif out_path.endswith(".gbk.txt"):
        stem = out_path[:-len(".gbk.txt")]
    elif out_path.endswith(".txt"):
        stem = out_path[:-len(".txt")]
    else:
        stem = out_path
    return f"{stem}.utf8.txt", f"{stem}.gbk.txt"


class TxtExporter(ExporterBase):
    """纯文本导出：页标记 + 纯文本块，分别落 UTF-8/GBK。"""

    def export(self, snapshot: ExportProjectSnapshot, out_path: str) -> None:
        document = build_export_ir(snapshot, "txt")
        page_sections: list[str] = []
        for page in document.pages:
            block_texts: list[str] = []
            for element in page.elements:
                texts = _txt_element_lines(element)
                if not texts:
                    continue
                block_texts.append("\n".join(texts))
            section = f"=== 第 {page.page_number} 页 ==="
            if block_texts:
                section += "\n" + "\n\n".join(block_texts)
            page_sections.append(section)
        content = "\n\n".join(page_sections).rstrip() + ("\n" if page_sections else "")
        utf8_path, gbk_path = txt_output_paths(out_path)
        with open(utf8_path, "w", encoding="utf-8") as f:
            f.write(content)
        with open(gbk_path, "w", encoding="gbk") as f:
            f.write(content)


def _txt_element_lines(element: ExportElement) -> list[str]:
    lines = element_lines(element)
    if element.kind == "paragraph":
        text = join_reflow_text_lines(lines).strip()
        return [text] if text else []
    return lines
