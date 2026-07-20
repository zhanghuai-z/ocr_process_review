"""Export IR JSON renderer."""
from __future__ import annotations

import json

from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.models.export_snapshot import ExportProjectSnapshot


class JsonExporter(ExporterBase):
    """Write the typed Export IR document itself."""

    def export(self, snapshot: ExportProjectSnapshot, out_path: str) -> None:
        document = build_export_ir(snapshot, "json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")
