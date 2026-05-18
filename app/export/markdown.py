"""Markdown 导出：保留页、块、行层次的可读文本。"""
from __future__ import annotations

from app.export.base import ExporterBase
from app.models import Block, BlockType, OcrProject
from app.services.export_service import (
    format_bbox,
    get_block_label,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)


def _one_line(text: str) -> str:
    return " ".join((text or "").splitlines()).strip()


def _heading_text(text: str) -> str:
    text = _one_line(text)
    if not text:
        return "未命名标题"
    return text.replace("#", r"\#")


def _block_comment(block: Block) -> str:
    return (
        f'<!-- block type="{block.block_type.value}" label="{get_block_label(block)}" '
        f'order="{block.order}" bbox="{format_bbox(block.bbox)}" -->'
    )


def _line_comment(line) -> str:
    return (
        f'<!-- line bbox="{format_bbox(line.bbox)}" '
        f'confidence="{line.confidence:.4f}" status="{line.proof_status.value}" -->'
    )


class MarkdownExporter(ExporterBase):
    """Markdown 初版规则：页为二级标题，块用注释保留元数据，行保持原顺序。"""

    def export(self, project: OcrProject, out_path: str) -> None:
        parts: list[str] = [
            f"# {_heading_text(project.name)}",
            "",
            f"<!-- pages={project.page_count} lines={project.total_lines} -->",
            "",
        ]

        for page in iter_export_pages(project):
            parts.extend([
                f"## 第 {page.page_number} 页",
                "",
                f'<!-- image="{page.image_path}" size="{page.width}x{page.height}" -->',
                "",
            ])
            for block in iter_export_blocks(page):
                parts.append(_block_comment(block))
                parts.extend(self._render_block(block))
                parts.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts).rstrip() + "\n")

    def _render_block(self, block: Block) -> list[str]:
        line_texts = [
            get_export_text(line)
            for line in iter_export_lines(block)
            if get_export_text(line).strip()
        ]
        if not line_texts:
            if block.note:
                return [f"> [{get_block_label(block)}] {_one_line(block.note)}"]
            return []

        if block.block_type == BlockType.TITLE:
            return [f"### {_heading_text(text)}" for text in line_texts]

        if block.block_type in (BlockType.FIGURE_CAPTION, BlockType.TABLE_CAPTION):
            label = get_block_label(block)
            return [f"*{label}：{_one_line(text)}*" for text in line_texts]

        if block.block_type == BlockType.EQUATION:
            body = "\n".join(_one_line(text) for text in line_texts)
            return ["$$", body, "$$"]

        if block.block_type in (BlockType.FIGURE, BlockType.TABLE):
            label = get_block_label(block)
            return [f"**[{label}]**", *self._render_lines(block)]

        if block.block_type == BlockType.REFERENCE:
            return ["### 参考文献", *self._render_lines(block)]

        return self._render_lines(block)

    def _render_lines(self, block: Block) -> list[str]:
        rendered: list[str] = []
        for line in iter_export_lines(block):
            text = get_export_text(line).strip()
            if not text:
                continue
            rendered.append(_line_comment(line))
            rendered.append(text)
        return rendered
