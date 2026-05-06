"""导出服务：统一导出文本来源、导出前状态检查。"""
from __future__ import annotations
from pathlib import Path
from typing import List

from app.models import Line, OcrProject


def get_export_text(line: Line) -> str:
    """获取导出的最终文本。

    统一规则：
    - 永远读取人工最终文本 (line.text)
    - 不读取 llm_suggestion（除非人工已接受）
    - 不读取 ocr_text（除非人工未修改且没有原始文本）
    """
    return line.text


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
