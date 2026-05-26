"""Runtime JSON rule source for Export IR v1."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.export.ir import EXPORT_IR_VERSION, ExportProfile


_RULE_PATH = Path(__file__).parent / "rules" / "export_ir_v1.json"
_REQUIRED_KINDS = {
    "title",
    "paragraph",
    "reference",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "equation",
    "unknown",
}
_REQUIRED_FORMATS = {
    "txt",
    "rtf",
    "docx",
    "html",
    "md",
    "xml",
    "json",
    "pdf-single",
    "pdf-dual",
}


@dataclass(frozen=True)
class ExportRules:
    version: str
    formats: dict[str, dict[str, Any]]
    archive: dict[str, Any]
    block_type_to_kind: dict[str, str]
    kind_rules: dict[str, dict[str, Any]]
    fallback_strategies: dict[str, dict[str, Any]]

    def profile_for(self, fmt: str) -> ExportProfile:
        normalized = normalize_export_format(fmt)
        data = self.formats[normalized]
        option_keys = set(data) - {
            "mode",
            "include_assets",
            "include_diagnostics",
            "archive_role",
            "authority",
            "parity_group",
        }
        return ExportProfile(
            format=normalized,
            mode=str(data["mode"]),
            include_assets=bool(data.get("include_assets", True)),
            include_diagnostics=bool(data.get("include_diagnostics", True)),
            archive_role=str(data.get("archive_role") or ""),
            authority=str(data.get("authority") or ""),
            parity_group=str(data.get("parity_group") or ""),
            options={key: data[key] for key in sorted(option_keys)},
        )

    def kind_for_block_type(self, block_type: str) -> str:
        return self.block_type_to_kind.get(block_type, "unknown")

    def rule_for_kind(self, kind: str) -> dict[str, Any]:
        return self.kind_rules.get(kind, self.kind_rules["unknown"])

    def fallback_for_kind(self, kind: str) -> dict[str, Any]:
        fallback_name = self.rule_for_kind(kind).get("fallback")
        if not fallback_name:
            return {}
        return self.fallback_strategies.get(fallback_name, {})


def normalize_export_format(fmt: str) -> str:
    normalized = fmt.lower().strip().lstrip(".")
    if normalized == "markdown":
        return "md"
    if normalized == "pdf":
        return "pdf-single"
    return normalized


def load_export_rules(path: str | Path | None = None) -> ExportRules:
    rule_path = Path(path) if path is not None else _RULE_PATH
    with open(rule_path, encoding="utf-8") as f:
        raw = json.load(f)
    return validate_export_rules(raw)


def validate_export_rules(raw: dict[str, Any]) -> ExportRules:
    version = raw.get("version")
    if version != EXPORT_IR_VERSION:
        raise ValueError(f"unsupported export rule version: {version!r}")

    formats = raw.get("formats")
    archive = raw.get("archive")
    block_type_to_kind = raw.get("block_type_to_kind")
    kind_rules = raw.get("kind_rules")
    fallback_strategies = raw.get("fallback_strategies")
    if not isinstance(formats, dict) or not isinstance(kind_rules, dict):
        raise ValueError("export rules must contain object fields: formats, kind_rules")
    if not isinstance(block_type_to_kind, dict):
        raise ValueError("export rules must contain block_type_to_kind")
    if not isinstance(archive, dict):
        raise ValueError("export rules must contain archive")
    if not isinstance(fallback_strategies, dict):
        raise ValueError("export rules must contain fallback_strategies")

    missing_formats = _REQUIRED_FORMATS - set(formats)
    if missing_formats:
        raise ValueError(f"export rules missing formats: {sorted(missing_formats)}")
    missing_kinds = _REQUIRED_KINDS - set(kind_rules)
    if missing_kinds:
        raise ValueError(f"export rules missing kinds: {sorted(missing_kinds)}")
    mapped_kinds = set(block_type_to_kind.values())
    missing_mapped_kinds = mapped_kinds - set(kind_rules)
    if missing_mapped_kinds:
        raise ValueError(f"block_type_to_kind references missing kinds: {sorted(missing_mapped_kinds)}")

    for fmt, profile in formats.items():
        if "mode" not in profile:
            raise ValueError(f"export profile {fmt!r} missing mode")
    if archive.get("authority_format") != "xml" or archive.get("mirror_format") != "json":
        raise ValueError("archive authority/mirror must be xml/json for v1")
    if not archive.get("mandatory_parity"):
        raise ValueError("export rules must declare mandatory_parity")

    for kind, rule in kind_rules.items():
        fallback_name = rule.get("fallback")
        if fallback_name and fallback_name not in fallback_strategies:
            raise ValueError(f"kind {kind!r} references missing fallback {fallback_name!r}")

    return ExportRules(
        version=version,
        formats=formats,
        archive=archive,
        block_type_to_kind=block_type_to_kind,
        kind_rules=kind_rules,
        fallback_strategies=fallback_strategies,
    )
