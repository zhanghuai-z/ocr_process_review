"""Typed Export IR v1 contracts.

The project model remains the source of truth.  These dataclasses are the
normalised export boundary consumed by format renderers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EXPORT_IR_VERSION = "export_ir.v1"

ExportFormat = Literal[
    "txt",
    "rtf",
    "docx",
    "html",
    "xml",
    "json",
    "pdf-single",
    "pdf-dual",
]

ExportKind = Literal[
    "title",
    "paragraph",
    "reference",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "equation",
    "unknown",
]


@dataclass(frozen=True)
class ExportProfile:
    format: str
    mode: str
    include_assets: bool = True
    include_diagnostics: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportProjectMeta:
    name: str
    page_count: int
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None


@dataclass(frozen=True)
class ExportSource:
    page_number: int
    block_ids: list[int | str]
    origin: str
    line_ids: list[int | str] = field(default_factory=list)
    char_ids: list[int | str] = field(default_factory=list)
    block_type: str = ""


@dataclass(frozen=True)
class ExportProof:
    status: str = "unchecked"
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    corrected: bool = False


@dataclass(frozen=True)
class ExportFallback:
    used: bool
    reason: str
    mode: str
    asset_ref: str | None = None


@dataclass(frozen=True)
class ExportElement:
    id: str
    kind: str
    page: int
    order: int
    source: ExportSource
    payload: dict[str, Any]
    bbox: dict[str, int] | None = None
    proof: ExportProof | None = None
    fallback: ExportFallback | None = None


@dataclass(frozen=True)
class ExportPage:
    page_number: int
    source_image: str
    size: dict[str, int]
    elements: list[ExportElement]
    source_meta: dict[str, Any] = field(default_factory=dict)
    status: str = ""


@dataclass(frozen=True)
class ExportAsset:
    id: str
    kind: str
    page_number: int
    bbox: dict[str, int] | None = None
    path: str = ""
    mime: str = ""


@dataclass(frozen=True)
class ExportDiagnostic:
    level: str
    code: str
    message: str
    element_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportDocument:
    version: str
    profile: ExportProfile
    project: ExportProjectMeta
    pages: list[ExportPage]
    assets: list[ExportAsset] = field(default_factory=list)
    diagnostics: list[ExportDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned_item = _drop_empty(item)
            if cleaned_item in ({}, []) and key not in ("assets", "diagnostics", "elements"):
                continue
            cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [_drop_empty(item) for item in value]
    return value
