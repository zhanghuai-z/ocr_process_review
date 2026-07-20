"""Export adapter contract."""
from __future__ import annotations
from abc import ABC, abstractmethod

from app.models.export_snapshot import ExportProjectSnapshot


class ExporterBase(ABC):
    """所有导出器实现此接口。"""

    @abstractmethod
    def export(self, snapshot: ExportProjectSnapshot, out_path: str) -> None:
        """Export one immutable transaction snapshot to the target path."""
        ...
