"""Fake LLM 预审引擎：用于测试。

输入固定文本，输出固定建议。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from app.models import LlmReviewStatus


@dataclass
class LlmPreReviewLine:
    """LLM 预审输入行。"""
    page_number: int
    block_order: int
    line_index: int
    text: str
    confidence: float
    block_type: str = "text"
    previous_text: Optional[str] = None
    next_text: Optional[str] = None


@dataclass
class LlmPreReviewSuggestion:
    """LLM 预审输出建议。"""
    page_number: int
    block_order: int
    line_index: int
    original_text: str
    suggested_text: str
    confidence: float
    reason: str
    flags: List[str] = field(default_factory=list)


@dataclass
class LlmPreReviewOptions:
    """LLM 预审配置选项。"""
    batch_size: int = 30
    send_context: bool = True
    send_page_text_only: bool = True


class FakeLlmPreReviewEngine:
    """Fake LLM 预审引擎。

    规则：
    - `错別字` → 建议 `错别字`，flag=ocr_typo
    - 空文本 → flag=empty_text
    - confidence < 0.8 → flag=low_confidence
    - 其他 → 返回原文
    """

    def review_lines(
        self,
        lines: List[LlmPreReviewLine],
        options: Optional[LlmPreReviewOptions] = None,
    ) -> List[LlmPreReviewSuggestion]:
        suggestions = []
        for line in lines:
            flags = []
            suggested = line.text
            reason = ""

            if not line.text.strip():
                flags.append("empty_text")
                reason = "文本为空"
            elif "错別字" in line.text:
                suggested = line.text.replace("错別字", "错别字")
                flags.append("ocr_typo")
                reason = "疑似错别字"
            elif line.confidence < 0.8:
                flags.append("low_confidence")
                reason = "置信度偏低"

            if not reason:
                reason = "未发现明确错误"

            suggestions.append(LlmPreReviewSuggestion(
                page_number=line.page_number,
                block_order=line.block_order,
                line_index=line.line_index,
                original_text=line.text,
                suggested_text=suggested,
                confidence=line.confidence,
                reason=reason,
                flags=flags,
            ))

        return suggestions
