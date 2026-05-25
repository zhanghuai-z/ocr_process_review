"""Markdown 导出：保留页、块、行层次的可读文本。"""
from __future__ import annotations

from app.export.base import ExporterBase
from app.export.ir import ExportElement
from app.export.ir_builder import build_export_ir
from app.export.rendering import bbox_attr, element_label, element_lines
from app.models import OcrProject


def _one_line(text: str) -> str:
    return " ".join((text or "").splitlines()).strip()


def _heading_text(text: str) -> str:
    text = _one_line(text)
    if not text:
        return "未命名标题"
    return text.replace("#", r"\#")


def _block_comment(element: ExportElement) -> str:
    return (
        f'<!-- block type="{element.kind}" label="{element_label(element)}" '
        f'order="{element.order}" bbox="{bbox_attr(element.bbox)}" -->'
    )


def _line_comment(item: dict) -> str:
    bbox = item.get("bbox") or {}
    return (
        f'<!-- line bbox="{bbox_attr(bbox)}" '
        f'confidence="{float(item.get("confidence") or 0):.4f}" status="{item.get("status") or ""}" -->'
    )


class MarkdownExporter(ExporterBase):
    """Markdown 初版规则：页为二级标题，块用注释保留元数据，行保持原顺序。"""

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "md")
        parts: list[str] = [
            f"# {_heading_text(document.project.name)}",
            "",
            f"<!-- pages={document.project.page_count} lines={document.project.summary.get('total_lines', 0)} -->",
            "",
        ]

        for page in document.pages:
            parts.extend([
                f"## 第 {page.page_number} 页",
                "",
                f'<!-- image="{page.source_image}" size="{page.size["w"]}x{page.size["h"]}" -->',
                "",
            ])
            for element in page.elements:
                parts.append(_block_comment(element))
                parts.extend(self._render_element(element))
                parts.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts).rstrip() + "\n")

    def _render_element(self, element: ExportElement) -> list[str]:
        line_texts = [text for text in element_lines(element) if text.strip()]
        if not line_texts:
            return []

        if element.kind == "title":
            return [f"### {_heading_text(text)}" for text in line_texts]

        if element.kind in {"figure_caption", "table_caption"}:
            label = element_label(element)
            return [f"*{label}：{_one_line(text)}*" for text in line_texts]

        if element.kind == "equation":
            body = "\n".join(_one_line(text) for text in line_texts)
            return ["$$", body, "$$"]

        if element.kind in {"figure", "table"}:
            label = element_label(element)
            return [f"**[{label}]**", *self._render_lines(element)]

        if element.kind == "reference":
            return ["### 参考文献", *self._render_lines(element)]

        return self._render_lines(element)

    def _render_lines(self, element: ExportElement) -> list[str]:
        rendered: list[str] = []
        payload_lines = element.payload.get("lines") or []
        if payload_lines:
            source = [(str(item.get("text") or ""), item) for item in payload_lines]
        else:
            source = [(text, {}) for text in element_lines(element)]
        for text, item in source:
            text = text.strip()
            if not text:
                continue
            rendered.append(_line_comment(item))
            rendered.append(text)
        return rendered
