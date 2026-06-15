"""Export format settings.

The UI is still intentionally small, but each export format already has a
dedicated settings slot so format-specific policy does not leak into renderers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MarkdownTableStyle = Literal[
    "three_line_html",
    "source_html",
    "plain_html",
    "image_fallback",
]


@dataclass(frozen=True)
class TxtExportSettings:
    pass


@dataclass(frozen=True)
class JsonExportSettings:
    pass


@dataclass(frozen=True)
class XmlExportSettings:
    pass


@dataclass(frozen=True)
class HtmlExportSettings:
    pass


@dataclass(frozen=True)
class DocxExportSettings:
    pass


@dataclass(frozen=True)
class RtfExportSettings:
    pass


@dataclass(frozen=True)
class PdfExportSettings:
    pass


@dataclass(frozen=True)
class MarkdownExportSettings:
    """Markdown-specific export policy.

    ``table_header_rows`` is optional by design: ``None`` means derive it from
    table structure, while a later UI can expose concrete presets per document
    type without changing the renderer contract.
    """

    filtered_source_labels: frozenset[str] = field(default_factory=lambda: frozenset({
        "header",
        "page_number",
        "number",
    }))
    merge_adjacent_captions: bool = True
    merge_equation_numbers: bool = True
    unwrap_math_delimiters: bool = True
    table_style: MarkdownTableStyle = "three_line_html"
    table_header_rows: int | None = None
    max_auto_table_header_rows: int = 4


@dataclass(frozen=True)
class ExportSettings:
    txt: TxtExportSettings = field(default_factory=TxtExportSettings)
    json: JsonExportSettings = field(default_factory=JsonExportSettings)
    xml: XmlExportSettings = field(default_factory=XmlExportSettings)
    html: HtmlExportSettings = field(default_factory=HtmlExportSettings)
    docx: DocxExportSettings = field(default_factory=DocxExportSettings)
    rtf: RtfExportSettings = field(default_factory=RtfExportSettings)
    pdf_single: PdfExportSettings = field(default_factory=PdfExportSettings)
    pdf_dual: PdfExportSettings = field(default_factory=PdfExportSettings)
    markdown: MarkdownExportSettings = field(default_factory=MarkdownExportSettings)


DEFAULT_EXPORT_SETTINGS = ExportSettings()
