"""Markdown light semantic reflow export."""
from __future__ import annotations

import os
import re
from html import escape as html_escape
from pathlib import Path

from app.export.base import ExporterBase
from app.export.ir import ExportAsset, ExportElement
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines, join_reflow_text_lines
from app.models import OcrProject


class MarkdownExporter(ExporterBase):
    """Markdown-native projection for the bounded light-reflow contract."""

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "md")
        out_dir = Path(out_path).parent
        assets = {asset.id: asset for asset in document.assets}
        parts: list[str] = []
        for page in document.pages:
            for element in page.elements:
                rendered = _render_element(element, assets, out_dir)
                if rendered:
                    parts.extend(rendered)
                    parts.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts).rstrip() + ("\n" if parts else ""))


def _render_element(element: ExportElement, assets: dict[str, ExportAsset], out_dir: Path) -> list[str]:
    if element.kind == "title":
        return [f"{'#' * _heading_level(element)} {_escape_markdown(_one_line(text))}" for text in element_lines(element)]
    if element.kind == "paragraph":
        text = join_reflow_text_lines(element_lines(element)).strip()
        return [_escape_markdown(text)] if text else []
    if element.kind == "reference":
        return [_escape_markdown(_one_line(text)) for text in element_lines(element)]
    if element.kind in {"figure_caption", "table_caption"}:
        return [f"*{_escape_markdown(_one_line(text))}*" for text in element_lines(element)]
    if element.kind == "figure":
        return _render_figure(element, assets, out_dir)
    if element.kind == "table":
        return _render_table(element)
    if element.kind == "equation":
        return _render_equation(element, assets, out_dir)
    if element.kind == "unknown":
        return _render_unknown(element)
    return [_escape_markdown(_one_line(text)) for text in element_lines(element)]


def _render_figure(element: ExportElement, assets: dict[str, ExportAsset], out_dir: Path) -> list[str]:
    payload = element.payload
    asset = assets.get(str(payload.get("asset_ref") or ""))
    alt_text = _one_line(str(payload.get("alt_text") or "figure"))
    if asset and asset.path:
        return [f"![{_escape_link_text(alt_text)}]({_relative_asset_path(asset.path, out_dir)})"]
    return [f"![{_escape_link_text(alt_text)}]()"]


def _render_table(element: ExportElement) -> list[str]:
    rows = element_lines(element)
    if not rows:
        rows = [_one_line(str(element.payload.get("mode") or "table"))]
    rendered = ["<table>"]
    for row in rows:
        rendered.append(f"  <tr><td>{html_escape(_one_line(row), quote=True)}</td></tr>")
    rendered.append("</table>")
    return rendered


def _render_equation(element: ExportElement, assets: dict[str, ExportAsset], out_dir: Path) -> list[str]:
    latex = _one_line(str(element.payload.get("latex") or ""))
    payload_lines = element.payload.get("lines") or []
    line_texts: list[str] = []
    for item in payload_lines:
        text = _one_line(str(item.get("text") or ""))
        if text:
            line_texts.append(text)
    body = latex or "\n".join(line_texts)
    if body.strip():
        return ["$$", body.strip(), "$$"]
    asset = assets.get(str(element.payload.get("asset_ref") or ""))
    if asset and asset.path:
        return [f"![equation]({_relative_asset_path(asset.path, out_dir)})"]
    return []


def _render_unknown(element: ExportElement) -> list[str]:
    text = " ".join(_one_line(line) for line in element_lines(element)).strip()
    if not text:
        text = "Unsupported or unknown OCR block"
    return [
        '<div style="background-color:#fff3cd;border:1px solid #ffeeba;padding:8px;">',
        html_escape(text, quote=True),
        "</div>",
    ]


def _heading_level(element: ExportElement) -> int:
    try:
        level = int(element.payload.get("heading_level") or 1)
    except (TypeError, ValueError):
        level = 1
    return max(1, min(6, level))


def _relative_asset_path(path: str, out_dir: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(out_dir.resolve())
    except ValueError:
        rel = Path(os.path.relpath(path, out_dir))
    return rel.as_posix()


def _one_line(text: str) -> str:
    return " ".join((text or "").splitlines()).strip()


def _escape_markdown(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    for ch in ("`", "[", "]", "*", "_"):
        escaped = escaped.replace(ch, f"\\{ch}")
    escaped = re.sub(r"^(\d+)([.)])(\s+)", r"\1\\\2\3", escaped)
    if escaped.startswith(("#", ">", "-", "+", "*")):
        escaped = "\\" + escaped
    return escaped


def _escape_link_text(text: str) -> str:
    return _escape_markdown(text).replace("!", r"\!")
