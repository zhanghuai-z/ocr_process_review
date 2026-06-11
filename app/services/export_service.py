"""导出服务：统一导出文本来源、结构顺序和导出前状态检查。

结构化导出初版口径：
- XML / HTML / Markdown / TXT 都按项目 -> 页 -> 块 -> 行读取；
- 行文本统一读取人工最终文本 line.final_text；
- 块按 order 优先、坐标兜底排序，尽量贴近原稿阅读顺序；
- XML / HTML 保留空块结构，TXT / Markdown 默认只输出有文本的块。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List

from app.core.block_attributes import block_attributes, semantic_block_type
from app.models import Block, BlockType, Line, OcrProject, Page


BLOCK_LABELS: dict[BlockType, str] = {
    BlockType.TEXT: "正文",
    BlockType.TITLE: "标题",
    BlockType.FIGURE: "图片",
    BlockType.FIGURE_CAPTION: "图注",
    BlockType.TABLE: "表格",
    BlockType.TABLE_CAPTION: "表注",
    BlockType.REFERENCE: "参考文献",
    BlockType.EQUATION: "公式",
    BlockType.UNKNOWN: "未知块",
}

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class ExportBlockStyle:
    """默认导出样式预设中的块样式。"""
    name: str
    html_class: str
    docx_style: str
    font_size_pt: int
    line_height_mm: int
    italic: bool = False


DEFAULT_STYLE_PRESET: dict[BlockType, ExportBlockStyle] = {
    BlockType.TITLE: ExportBlockStyle("标题", "block-title", "Heading 2", 16, 10),
    BlockType.TEXT: ExportBlockStyle("正文", "block-body", "Normal", 12, 8),
    BlockType.REFERENCE: ExportBlockStyle("参考文献", "block-reference", "Normal", 11, 7),
    BlockType.FIGURE_CAPTION: ExportBlockStyle("图注", "block-caption", "Caption", 10, 6, italic=True),
    BlockType.TABLE_CAPTION: ExportBlockStyle("表注", "block-caption", "Caption", 10, 6, italic=True),
    BlockType.EQUATION: ExportBlockStyle("公式", "block-equation", "Normal", 12, 8),
    BlockType.TABLE: ExportBlockStyle("表格", "block-table", "Normal", 11, 7),
    BlockType.FIGURE: ExportBlockStyle("图片", "block-figure", "Normal", 11, 7),
    BlockType.UNKNOWN: ExportBlockStyle("未知块", "block-unknown", "Normal", 12, 8),
}


def get_block_style(block: Block) -> ExportBlockStyle:
    """返回当前块在 HTML/DOCX/PDF/RTF 中共享的默认样式。"""
    block_type = semantic_block_type(block)
    return DEFAULT_STYLE_PRESET.get(block_type, DEFAULT_STYLE_PRESET[BlockType.UNKNOWN])


def get_export_text(line: Line) -> str:
    """获取导出的最终文本。

    统一规则：
    - 永远读取人工最终文本 (line.final_text)
    - 不读取 llm_suggestion（除非人工已接受）
    - 不读取 ocr_text（除非人工未修改且没有原始文本）
    """
    return line.display_text


def get_block_label(block: Block) -> str:
    """返回导出时使用的人类可读块类型。"""
    attrs = block_attributes(block)
    base = BLOCK_LABELS.get(attrs.semantic_block_type, attrs.semantic_block_type.value)
    semantic = attrs.normalized_semantic_label
    if semantic and semantic != attrs.semantic_block_type.value:
        return f"{base}（{semantic}）"
    return base


def format_bbox(bbox) -> str:
    """将 bbox 格式化为稳定的 x,y,w,h 字符串。"""
    return f"{bbox.x},{bbox.y},{bbox.w},{bbox.h}"


def sanitize_export_filename(name: str, fallback: str = "ocr_export") -> str:
    """生成跨平台安全的导出文件名主体。"""
    cleaned = _INVALID_FILENAME_CHARS.sub("_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    if not cleaned:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned[:120]


def build_export_path(out_dir: str | Path, project_name: str, fmt: str) -> Path:
    """根据项目名和格式生成安全落盘路径，并确保输出目录存在。"""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    normalized = fmt.lower().strip().lstrip(".")
    base = sanitize_export_filename(project_name)
    if normalized == "txt":
        return directory / f"{base}.utf8.txt"
    if normalized in {"pdf-single", "pdf-dual"}:
        return directory / f"{base}.{normalized}.pdf"
    suffix = {
        "markdown": "md",
        "json": "json",
        "pdf": "pdf",
    }.get(normalized, normalized)
    return directory / f"{base}.{suffix}"


def iter_export_pages(project: OcrProject) -> Iterable[Page]:
    """按页码输出页面，页码缺失时保持原列表顺序。"""
    return sorted(project.pages, key=lambda page: page.page_number)


def iter_export_blocks(page: Page, *, include_empty: bool = False) -> Iterable[Block]:
    """按统一阅读顺序输出块。"""
    blocks = sorted(
        page.blocks,
        key=lambda block: (block.order, block.bbox.y, block.bbox.x),
    )
    for block in blocks:
        if include_empty or block.lines or block.note:
            yield block


def iter_export_lines(block: Block) -> Iterable[Line]:
    """输出块内行，统一文本来源由 get_export_text 控制。"""
    return block.lines


def check_export_readiness(project: OcrProject) -> List[str]:
    """检查导出就绪状态，返回警告列表。"""
    warnings = []
    summary = project.get_export_summary()

    if not project.pages:
        warnings.append("项目中没有页面")
        return warnings

    if summary["unrecognized_blocks"] > 0:
        warnings.append(
            f"{summary['unrecognized_blocks']} 个块尚未 OCR 识别"
        )

    if summary["unproofed_lines"] > 0:
        warnings.append(
            f"{summary['unproofed_lines']} 行尚未校对确认"
        )

    if summary["flagged_lines"] > 0:
        warnings.append(
            f"{summary['flagged_lines']} 行存在低置信度标记"
        )

    return warnings
