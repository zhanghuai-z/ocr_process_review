from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from tools.ocr_inspector.models import DocumentNode


class BaseAdapter(ABC):
    @abstractmethod
    def parse(self, raw: Any, *, source_path: str = "", image_path: str = "") -> DocumentNode:
        ...

    @classmethod
    def detect(cls, raw: Any) -> bool:
        return False


class AdapterRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[BaseAdapter]) -> None:
        self._registry[name] = adapter_cls

    def get(self, name: str) -> Optional[Type[BaseAdapter]]:
        return self._registry.get(name)

    def auto_detect(self, raw: Any) -> Optional[Type[BaseAdapter]]:
        for cls in self._registry.values():
            if cls.detect(raw):
                return cls
        return None

    def names(self) -> list:
        return list(self._registry.keys())
