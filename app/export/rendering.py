"""Shared helpers for rendering Export IR."""
from __future__ import annotations

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
        return str(payload.get("latex") or payload.get("text") or f"[公式: {payload.get('mode', 'image_fallback')}]")
    return f"[{element_label(element)}]"


def element_lines(element: ExportElement) -> list[str]:
    lines = element.payload.get("lines") or []
    if lines:
        return [str(item.get("text") or "") for item in lines if str(item.get("text") or "").strip()]
    text = element_text(element)
    return [text] if text.strip() else []


def bbox_attr(bbox: dict[str, int] | None) -> str:
    if not bbox:
        return ""
    return f"{bbox.get('x', 0)},{bbox.get('y', 0)},{bbox.get('w', 0)},{bbox.get('h', 0)}"


def html_escape(text: str) -> str:
    return escape(text, quote=True)
