from .base import BaseAdapter, AdapterRegistry
from .paddle import PaddleAdapter

registry = AdapterRegistry()
registry.register("paddle", PaddleAdapter)

__all__ = ["BaseAdapter", "AdapterRegistry", "PaddleAdapter", "registry"]
