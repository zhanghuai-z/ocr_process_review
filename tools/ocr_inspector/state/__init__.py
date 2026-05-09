"""Application state for OCR Inspector."""
from __future__ import annotations
from typing import Any, Callable, Optional
from tools.ocr_inspector.models.ir import BlockNode, CharNode, DocumentNode, LineNode, PageNode

IrNode = Any


class AppState:
    def __init__(self) -> None:
        self._document: Optional[DocumentNode] = None
        self._active_page: Optional[PageNode] = None
        self._selected_node: Optional[IrNode] = None
        self._overlay_flags: dict[str, bool] = {
            "blocks": True, "lines": True, "chars": False,
            "polygons": False, "labels": True,
        }
        self._on_document_changed: list[Callable] = []
        self._on_page_changed: list[Callable] = []
        self._on_selection_changed: list[Callable] = []
        self._on_overlay_changed: list[Callable] = []

    @property
    def document(self) -> Optional[DocumentNode]:
        return self._document

    def set_document(self, doc: Optional[DocumentNode]) -> None:
        self._document = doc
        self._selected_node = None
        self._active_page = doc.pages[0] if doc and doc.pages else None
        for cb in self._on_document_changed:
            cb(doc)

    @property
    def active_page(self) -> Optional[PageNode]:
        return self._active_page

    def set_active_page(self, page: Optional[PageNode]) -> None:
        self._active_page = page
        self._selected_node = None
        for cb in self._on_page_changed:
            cb(page)

    @property
    def selected_node(self) -> Optional[IrNode]:
        return self._selected_node

    def set_selection(self, node: Optional[IrNode]) -> None:
        self._selected_node = node
        for cb in self._on_selection_changed:
            cb(node)

    @property
    def overlay_flags(self) -> dict[str, bool]:
        return dict(self._overlay_flags)

    def toggle_overlay(self, key: str) -> None:
        if key in self._overlay_flags:
            self._overlay_flags[key] = not self._overlay_flags[key]
            for cb in self._on_overlay_changed:
                cb(key, self._overlay_flags[key])

    def set_overlay(self, key: str, value: bool) -> None:
        if key in self._overlay_flags:
            self._overlay_flags[key] = value
            for cb in self._on_overlay_changed:
                cb(key, value)

    def on_document_changed(self, cb: Callable) -> None:
        self._on_document_changed.append(cb)

    def on_page_changed(self, cb: Callable) -> None:
        self._on_page_changed.append(cb)

    def on_selection_changed(self, cb: Callable) -> None:
        self._on_selection_changed.append(cb)

    def on_overlay_changed(self, cb: Callable) -> None:
        self._on_overlay_changed.append(cb)
