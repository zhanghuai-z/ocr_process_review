"""共享页面目录列表组件（Phase 24）。

来源：原 LayoutPanel 左侧页面缩略列表。横校面板需要"和版面分析使用同一个
页面目录"，故抽出为可复用 widget，由 LayoutPanel / HProofPanel 共用，避免
样式 / 行为分叉。

对外接口：
- ``set_pages(pages)`` 重建条目（每页一行："第 N 页 / 文件名"）
- ``set_current_index(idx)`` 同步选中行（不触发 page_selected）
- ``page_selected: Signal(int)`` 用户点击切换时派发当前 index
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget

from app.models import Page


class PageDirectoryList(QListWidget):
    """轻量子类，仅约束样式 / 信号契约，不引入额外业务状态。"""

    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(140)
        self.setMinimumWidth(70)
        self._suppress_signal = False
        self.currentRowChanged.connect(self._on_row_changed)

    # ── public ────────────────────────────────────────────────

    def set_pages(self, pages: List[Page]) -> None:
        self._suppress_signal = True
        try:
            self.clear()
            for page in pages:
                src = getattr(page, "source_path", None) or page.image_path or ""
                fname = Path(src).name if src else ""
                label = f"第 {page.page_number} 页"
                if fname:
                    label += f"\n{fname}"
                self.addItem(label)
        finally:
            self._suppress_signal = False

    def set_current_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.count():
            return
        self._suppress_signal = True
        try:
            self.setCurrentRow(idx)
        finally:
            self._suppress_signal = False

    # ── internal ──────────────────────────────────────────────

    def _on_row_changed(self, idx: int) -> None:
        if self._suppress_signal:
            return
        if 0 <= idx < self.count():
            self.page_selected.emit(idx)
