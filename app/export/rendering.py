"""Shared helpers for rendering Export IR."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

from app.export.ir import ExportDocument, ExportElement, ExportPage


KIND_LABELS: dict[str, str] = {
    "title": "标题",
    "paragraph": "正文",
    "reference": "参考文献",
    "figure": "图片",
    "figure_caption": "图注",
    "table": "表格",
    "table_caption": "表注",
    "equation": "公式",
    "unknown": "未知块",
}

KIND_HTML_CLASS: dict[str, str] = {
    "title": "block-title",
    "paragraph": "block-body",
    "reference": "block-reference",
    "figure": "block-figure",
    "figure_caption": "block-caption",
    "table": "block-table",
    "table_caption": "block-caption",
    "equation": "block-equation",
    "unknown": "block-unknown",
}

RICH_REFLOW_ROLE: dict[str, str] = {
    "title": "heading",
    "paragraph": "body",
    "reference": "reference",
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "equation": "equation",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class RichReflowBlock:
    page_number: int
    kind: str
    role: str
    order: int
    label: str
    html_class: str
    lines: list[str]
    proof_status: str = "unchecked"


def iter_ir_elements(document: ExportDocument) -> Iterable[tuple[ExportPage, ExportElement]]:
    for page in document.pages:
        for element in sorted(page.elements, key=lambda item: (item.order, item.bbox or {})):
            yield page, element


def element_label(element: ExportElement) -> str:
    return KIND_LABELS.get(element.kind, element.kind)


def element_text(element: ExportElement) -> str:
    payload = element.payload
    if "text" in payload:
        return str(payload.get("text") or "")
    if element.kind == "figure":
        alt_text = payload.get("alt_text") or ""
        return alt_text or f"[{element_label(element)}: {payload.get('asset_ref') or 'no-asset'}]"
    if element.kind == "table":
        text = str(payload.get("text") or "").strip()
        return text or f"[表格: {payload.get('mode', 'image_fallback')} {payload.get('asset_ref') or ''}]".strip()
    if element.kind == "equation":
        return str(payload.get("text") or f"[公式: {payload.get('mode', 'image_fallback')}]")
    return f"[{element_label(element)}]"


def element_lines(element: ExportElement) -> list[str]:
    lines = element.payload.get("lines") or []
    if lines:
        return [str(item.get("text") or "") for item in lines if str(item.get("text") or "").strip()]
    text = element_text(element)
    return [text] if text.strip() else []


def rich_reflow_lines(element: ExportElement) -> list[str]:
    lines = element_lines(element)
    if not lines:
        return []
    keep_linebreaks = bool(element.payload.get("keep_linebreaks"))
    if element.kind == "paragraph" and not keep_linebreaks:
        text = join_reflow_text_lines(lines).strip()
        return [text] if text else []
    return lines


def join_reflow_text_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if not line:
            continue
        if text and _needs_reflow_space(text[-1], line[0]):
            text += " "
        text += line
    return text


def _needs_reflow_space(left: str, right: str) -> bool:
    if left.isspace() or right.isspace():
        return False
    if not right.isascii() or not _is_reflow_word_start(right):
        return False
    if left.isascii():
        return _is_reflow_word_end(left) or left in ",.;:!?)]}"
    return True


def _is_reflow_word_start(ch: str) -> bool:
    return ch.isalnum()


def _is_reflow_word_end(ch: str) -> bool:
    return ch.isalnum()


def iter_rich_reflow_blocks(document: ExportDocument) -> Iterable[RichReflowBlock]:
    for page, element in iter_ir_elements(document):
        lines = rich_reflow_lines(element)
        if not lines:
            continue
        yield RichReflowBlock(
            page_number=page.page_number,
            kind=element.kind,
            role=RICH_REFLOW_ROLE.get(element.kind, "unknown"),
            order=element.order,
            label=element_label(element),
            html_class=KIND_HTML_CLASS.get(element.kind, "block-unknown"),
            lines=lines,
            proof_status=element.proof.status if element.proof else "unchecked",
        )


def rich_reflow_pages(document: ExportDocument) -> list[tuple[ExportPage, list[RichReflowBlock]]]:
    blocks_by_page: dict[int, list[RichReflowBlock]] = {}
    for block in iter_rich_reflow_blocks(document):
        blocks_by_page.setdefault(block.page_number, []).append(block)
    return [(page, blocks_by_page.get(page.page_number, [])) for page in document.pages]


def bbox_attr(bbox: dict[str, int] | None) -> str:
    if not bbox:
        return ""
    return f"{bbox.get('x', 0)},{bbox.get('y', 0)},{bbox.get('w', 0)},{bbox.get('h', 0)}"


def html_escape(text: str) -> str:
    return escape(text, quote=True)
