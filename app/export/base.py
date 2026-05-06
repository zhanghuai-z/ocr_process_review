"""导出器基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod

from app.models import OcrProject


class ExporterBase(ABC):
    """所有导出器实现此接口。"""

    @abstractmethod
    def export(self, project: OcrProject, out_path: str) -> None:
        """将项目导出到指定路径。"""
        ...
