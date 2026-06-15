"""Markdown light semantic reflow export."""
from __future__ import annotations

from dataclasses import replace
import os
import re
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path

from app.export.base import ExporterBase
from app.export.ir import ExportAsset, ExportElement
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines, join_reflow_text_lines
from app.export.settings import MarkdownExportSettings
from app.models import OcrProject


class MarkdownExporter(ExporterBase):
    """Markdown-native projection for the bounded light-reflow contract."""

    def __init__(self, settings: "MarkdownExportSettings | None" = None) -> None:
        self.settings = settings or MarkdownExportSettings()

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "md")
        out_file = Path(out_path)
        out_dir = out_file.parent
        asset_dir = out_dir / f"{out_file.stem}_assets"
        assets = {asset.id: asset for asset in document.assets}
        parts: list[str] = []
        for page in document.pages:
            for element in _prepare_markdown_elements(page.elements, self.settings):
                rendered = _render_element(element, assets, out_dir, asset_dir, self.settings)
                if rendered:
                    parts.extend(rendered)
                    parts.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts).rstrip() + ("\n" if parts else ""))


def _prepare_markdown_elements(
    elements: list[ExportElement],
    settings: MarkdownExportSettings,
) -> list[ExportElement]:
    visible = [
        element for element in elements
        if not _is_filtered_position_element(element, settings)
    ]
    prepared: list[ExportElement] = []
    index = 0
    while index < len(visible):
        element = visible[index]
        if (
            settings.merge_adjacent_captions
            and element.kind in {"figure_caption", "table_caption"}
        ):
            group = [element]
            index += 1
            while index < len(visible) and visible[index].kind == element.kind:
                group.append(visible[index])
                index += 1
            prepared.append(_merged_text_element(group))
            continue
        if settings.merge_equation_numbers and _is_equation_number(element):
            if index + 1 < len(visible) and visible[index + 1].kind == "equation":
                prepared.append(_equation_with_number(visible[index + 1], element, settings))
                index += 2
                continue
        if (
            settings.merge_equation_numbers
            and element.kind == "equation"
            and not _is_equation_number(element)
            and index + 1 < len(visible)
            and _is_equation_number(visible[index + 1])
        ):
            prepared.append(_equation_with_number(element, visible[index + 1], settings))
            index += 2
            continue
        prepared.append(element)
        index += 1
    return prepared


def _is_filtered_position_element(element: ExportElement, settings: MarkdownExportSettings) -> bool:
    labels = _element_labels(element)
    return any(label in settings.filtered_source_labels for label in labels)


def _element_labels(element: ExportElement) -> set[str]:
    attrs = element.layout_attributes or {}
    raw_payload = attrs.get("raw_payload") if isinstance(attrs, dict) else {}
    labels = {
        element.source.source_label,
        element.source.semantic_label,
        element.source.semantic_block_type,
        attrs.get("source_label") if isinstance(attrs, dict) else "",
        attrs.get("semantic_label") if isinstance(attrs, dict) else "",
        raw_payload.get("block_label") if isinstance(raw_payload, dict) else "",
        raw_payload.get("label") if isinstance(raw_payload, dict) else "",
        raw_payload.get("type") if isinstance(raw_payload, dict) else "",
    }
    return {_normalize_label(label) for label in labels if str(label or "").strip()}


def _normalize_label(label: object) -> str:
    return str(label or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_equation_number(element: ExportElement) -> bool:
    return element.kind == "equation" and bool(_element_labels(element) & {
        "formula_number",
        "equation_number",
    })


def _merged_text_element(elements: list[ExportElement]) -> ExportElement:
    if not elements:
        raise ValueError("cannot merge empty markdown element group")
    parts = [
        _one_line(text)
        for element in elements
        for text in element_lines(element)
        if _one_line(text)
    ]
    text = " ".join(parts).strip()
    payload = dict(elements[0].payload)
    payload["text"] = text
    payload["lines"] = [{"text": text}] if text else []
    return replace(elements[0], payload=payload)


def _equation_with_number(
    equation: ExportElement,
    number: ExportElement,
    settings: MarkdownExportSettings,
) -> ExportElement:
    body = _equation_body(equation, settings)
    number_text = _clean_equation_number(" ".join(element_lines(number)))
    if body and number_text:
        body = _append_equation_number(body, number_text)
    payload = dict(equation.payload)
    payload["text"] = body
    payload["lines"] = [{"text": body}] if body else []
    return replace(equation, payload=payload)


def _equation_body(element: ExportElement, settings: MarkdownExportSettings) -> str:
    latex = _one_line(str(element.payload.get("latex") or ""))
    payload_lines = element.payload.get("lines") or []
    line_texts: list[str] = []
    for item in payload_lines:
        text = _one_line(str(item.get("text") or ""))
        if text:
            line_texts.append(text)
    body = latex or " ".join(line_texts)
    if settings.unwrap_math_delimiters:
        body = _strip_math_delimiters(body)
    return body.strip()


def _strip_math_delimiters(text: str) -> str:
    value = _one_line(text)
    changed = True
    while changed:
        changed = False
        stripped = value.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
            value = stripped[2:-2].strip()
            changed = True
            continue
        if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
            value = stripped[1:-1].strip()
            changed = True
    return value.strip()


def _clean_equation_number(text: str) -> str:
    value = _strip_math_delimiters(text)
    value = value.strip()
    if len(value) >= 2 and (
        (value[0], value[-1]) in {("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")}
    ):
        value = value[1:-1].strip()
    return value


def _append_equation_number(body: str, number: str) -> str:
    if not number:
        return body
    if r"\tag{" in body:
        return body
    return f"{body} \\tag{{{number}}}"


def _render_element(
    element: ExportElement,
    assets: dict[str, ExportAsset],
    out_dir: Path,
    asset_dir: Path,
    settings: MarkdownExportSettings,
) -> list[str]:
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
        return _render_figure(element, assets, out_dir, asset_dir)
    if element.kind == "table":
        return _render_table(element, assets, out_dir, asset_dir, settings)
    if element.kind == "equation":
        return _render_equation(element, assets, out_dir, asset_dir, settings)
    if element.kind == "unknown":
        return _render_unknown(element, assets, out_dir, asset_dir)
    return [_escape_markdown(_one_line(text)) for text in element_lines(element)]


def _render_figure(
    element: ExportElement,
    assets: dict[str, ExportAsset],
    out_dir: Path,
    asset_dir: Path,
) -> list[str]:
    payload = element.payload
    asset = assets.get(str(payload.get("asset_ref") or ""))
    alt_text = _one_line(str(payload.get("alt_text") or "figure"))
    return [_render_asset_image(asset, out_dir, asset_dir, alt_text)]


def _render_table(
    element: ExportElement,
    assets: dict[str, ExportAsset],
    out_dir: Path,
    asset_dir: Path,
    settings: MarkdownExportSettings,
) -> list[str]:
    rows = element_lines(element)
    asset = assets.get(str(element.payload.get("asset_ref") or ""))
    if settings.table_style == "image_fallback":
        if asset:
            return [_render_asset_image(asset, out_dir, asset_dir, "table")]
        return _render_plain_html_table(rows)
    source_table = _raw_html_table(rows)
    if settings.table_style == "three_line_html":
        html_table = source_table or "\n".join(_render_plain_html_table(rows))
        styled = _render_three_line_table(html_table, settings)
        if styled:
            return [styled]
        if source_table:
            return [source_table]
    if source_table:
        return [source_table]
    if not rows:
        return [_render_asset_image(asset, out_dir, asset_dir, "table")]
    return _render_plain_html_table(rows)


def _render_equation(
    element: ExportElement,
    assets: dict[str, ExportAsset],
    out_dir: Path,
    asset_dir: Path,
    settings: MarkdownExportSettings,
) -> list[str]:
    body = _equation_body(element, settings)
    if body.strip():
        return ["$$", body.strip(), "$$"]
    asset = assets.get(str(element.payload.get("asset_ref") or ""))
    return [_render_asset_image(asset, out_dir, asset_dir, "equation")] if asset else []


def _raw_html_table(rows: list[str]) -> str:
    if not rows:
        return ""
    text = "\n".join(row.strip() for row in rows if row.strip()).strip()
    if not text:
        return ""
    candidate = html_unescape(text).strip()
    if re.match(r"(?is)^<table\b.*</table>$", candidate):
        return candidate
    return ""


def _render_plain_html_table(rows: list[str]) -> list[str]:
    if not rows:
        return []
    rendered = ["<table>"]
    for row in rows:
        rendered.append(f"  <tr><td>{html_escape(_one_line(row), quote=True)}</td></tr>")
    rendered.append("</table>")
    return rendered


def _render_three_line_table(table_html: str, settings: MarkdownExportSettings) -> str:
    if not table_html.strip():
        return ""
    try:
        from lxml import html

        table = html.fragment_fromstring(table_html, create_parent=False)
        if getattr(table, "tag", "").lower() != "table":
            tables = table.xpath(".//table")
            table = tables[0] if tables else table
        if getattr(table, "tag", "").lower() != "table":
            return ""

        _append_class(table, "ocr-three-line-table")
        _append_style(
            table,
            "border-collapse: collapse; margin: 16px auto; width: auto;",
        )
        rows = table.xpath(".//tr")
        if not rows:
            return html.tostring(table, encoding="unicode", method="html")

        for cell in table.xpath(".//th|.//td"):
            _append_style(
                cell,
                "border: none; padding: 4px 10px; text-align: center; vertical-align: middle;",
            )

        header_rows = _table_header_row_count(rows, settings)
        first_row = rows[0]
        header_row = rows[min(header_rows - 1, len(rows) - 1)]
        last_row = rows[-1]
        for cell in _row_cells(first_row):
            _append_style(cell, "border-top: 1.5px solid #000;")
        for cell in _row_cells(header_row):
            _append_style(cell, "border-bottom: 1px solid #000;")
        for row_index, row in enumerate(rows[:header_rows]):
            for cell in _row_cells(row):
                rowspan = _positive_int(cell.get("rowspan"), 1)
                if row_index < header_rows - 1 and row_index + rowspan >= header_rows:
                    _append_style(cell, "border-bottom: 1px solid #000;")
        for cell in _row_cells(last_row):
            _append_style(cell, "border-bottom: 1.5px solid #000;")
        return html.tostring(table, encoding="unicode", method="html")
    except Exception:
        return ""


def _append_class(element, class_name: str) -> None:
    current = str(element.get("class") or "").split()
    if class_name not in current:
        current.append(class_name)
    element.set("class", " ".join(current))


def _append_style(element, style: str) -> None:
    current = str(element.get("style") or "").strip()
    if current and not current.endswith(";"):
        current += ";"
    element.set("style", f"{current} {style}".strip())


def _row_cells(row) -> list:
    return list(row.xpath("./th|./td"))


def _table_header_row_count(rows: list, settings: MarkdownExportSettings) -> int:
    if not rows:
        return 0
    if settings.table_header_rows is not None:
        return max(1, min(int(settings.table_header_rows), len(rows)))
    header_rows = 1
    for row_index, row in enumerate(rows):
        if row_index >= max(1, settings.max_auto_table_header_rows):
            break
        for cell in _row_cells(row):
            colspan = _positive_int(cell.get("colspan"), 1)
            rowspan = _positive_int(cell.get("rowspan"), 1)
            if colspan > 1 or rowspan > 1:
                header_rows = max(header_rows, row_index + rowspan)
    return max(1, min(header_rows, len(rows), max(1, settings.max_auto_table_header_rows)))


def _positive_int(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _render_unknown(
    element: ExportElement,
    assets: dict[str, ExportAsset],
    out_dir: Path,
    asset_dir: Path,
) -> list[str]:
    text = " ".join(_one_line(line) for line in element_lines(element)).strip()
    if not text:
        asset = assets.get(str(element.payload.get("asset_ref") or ""))
        if asset:
            return [_render_asset_image(asset, out_dir, asset_dir, "unknown block")]
        text = "Unsupported or unknown OCR block"
    return [
        '<div style="background-color:#fff3cd;border:1px solid #ffeeba;padding:8px;">',
        html_escape(text, quote=True),
        "</div>",
    ]


def _render_asset_image(asset: ExportAsset | None, out_dir: Path, asset_dir: Path, alt_text: str) -> str:
    alt = _escape_link_text(_one_line(alt_text) or "asset")
    path = _materialize_asset_region(asset, out_dir, asset_dir)
    return f"![{alt}]({path})" if path else f"![{alt}]()"


def _materialize_asset_region(asset: ExportAsset | None, out_dir: Path, asset_dir: Path) -> str:
    if asset is None or not asset.path:
        return ""
    if not asset.bbox:
        return _relative_asset_path(asset.path, out_dir)
    source_path = Path(asset.path)
    if not source_path.exists():
        return _relative_asset_path(asset.path, out_dir)
    try:
        from PIL import Image

        with Image.open(source_path) as image:
            x, y, w, h = _clamped_crop_box(asset.bbox, image.width, image.height)
            if w <= 0 or h <= 0:
                return _relative_asset_path(asset.path, out_dir)
            asset_dir.mkdir(parents=True, exist_ok=True)
            target = asset_dir / f"{_safe_asset_name(asset.id)}_{_safe_asset_name(asset.kind)}.png"
            image.crop((x, y, x + w, y + h)).save(target)
            return _relative_asset_path(str(target), out_dir)
    except Exception:
        return _relative_asset_path(asset.path, out_dir)


def _clamped_crop_box(bbox: dict[str, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x = max(0, min(int(bbox.get("x") or 0), image_width))
    y = max(0, min(int(bbox.get("y") or 0), image_height))
    x2 = max(x, min(int((bbox.get("x") or 0) + (bbox.get("w") or 0)), image_width))
    y2 = max(y, min(int((bbox.get("y") or 0) + (bbox.get("h") or 0)), image_height))
    return x, y, x2 - x, y2 - y


def _safe_asset_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "asset")).strip("._") or "asset"


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
