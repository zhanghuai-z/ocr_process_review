"""导出服务：统一导出文本来源、结构顺序和导出前状态检查。

结构化导出初版口径：
- XML / HTML / Markdown / TXT 都按项目 -> 页 -> 块 -> 行读取；
- 行文本统一通过 proof_display_text 读取（OCR 文本或人工终稿的统一视图）；
- 块按 order 优先、坐标兜底排序，尽量贴近原稿阅读顺序；
- XML / HTML 保留空块结构，TXT / Markdown 默认只输出有文本的块。
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable, List

from app.core.proof_line_facts import proof_display_text
from app.models import Line, OcrProject, Page
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_observation import block_ocr_line_observations_by_uid
from app.services.ocr_dispatch_plan import build_text_ocr_dispatch_plan
from app.services.proof_stats_service import ProofStatsService


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def get_export_text(line: Line) -> str:
    """获取导出的最终文本。

    统一规则：
    - 永远通过 proof_display_text 读取当前校对文本
    - 不读取 ocr_text（除非人工未修改且没有原始文本）
    """
    return proof_display_text(line)


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


def iter_export_block_views(page: Page, *, include_empty: bool = False) -> Iterable[LayoutBlockView]:
    """Yield export blocks ordered by adopted layout snapshot facts."""
    views = sorted(
        (view for view in iter_page_layout_block_views(page) if view.runtime_block is not None),
        key=lambda view: (view.order, view.bbox.y, view.bbox.x),
    )
    for view in views:
        block = view.runtime_block
        if block is None:
            continue
        if include_empty or block_ocr_line_observations_by_uid(view.uid) or view.note:
            yield view


def iter_export_lines(view: LayoutBlockView) -> Iterable[Line]:
    """输出块内行，统一文本来源由 get_export_text 控制。"""
    return block_ocr_line_observations_by_uid(view.uid)


def build_export_summary(project: OcrProject) -> dict:
    """Return export readiness counters without asking models to interpret proof state."""
    proof_stats = ProofStatsService().summarize(project)
    unproofed = proof_stats.total_lines - proof_stats.confirmed_lines - proof_stats.modified_lines
    return {
        "total_pages": project.page_count,
        "total_lines": proof_stats.total_lines,
        "unproofed_lines": unproofed,
        "flagged_lines": proof_stats.flagged_lines,
        "unrecognized_blocks": sum(
            1
            for page in project.pages
            for target in build_text_ocr_dispatch_plan(page).text_blocks
            if not block_ocr_line_observations_by_uid(target.view.uid)
        ),
    }


def check_export_readiness(project: OcrProject) -> List[str]:
    """检查导出就绪状态，返回警告列表。"""
    warnings = []
    summary = build_export_summary(project)

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
