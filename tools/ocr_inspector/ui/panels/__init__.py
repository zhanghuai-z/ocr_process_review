"""JSON tree panel for OCR Inspector."""
from __future__ import annotations
from typing import Any, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QAbstractItemView

from tools.ocr_inspector.models.ir import BlockNode, CharNode, DocumentNode, LineNode, PageNode
from tools.ocr_inspector.state import AppState

_COL_LABEL, _COL_TEXT, _COL_CONF = 0, 1, 2


class JsonTreePanel(QTreeWidget):
    """Left pane: hierarchical IR document view."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._node_to_item: dict[int, QTreeWidgetItem] = {}

        self.setColumnCount(3)
        self.setHeaderLabels(["Node", "Text", "Conf"])
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 300)
        self.setColumnWidth(2, 60)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setUniformRowHeights(True)

        self.itemClicked.connect(self._on_item_clicked)
        state.on_document_changed(self._on_document_changed)
        state.on_selection_changed(self._on_external_selection)

    def _on_document_changed(self, doc: Optional[DocumentNode]) -> None:
        self.clear()
        self._node_to_item.clear()
        if doc is None:
            return
        self._build_document(doc)
        self.expandToDepth(1)

    def _build_document(self, doc: DocumentNode) -> None:
        root = self._make_item(None, "document", f"Document [{doc.engine}]",
                               doc.source_path, "", doc)
        self.addTopLevelItem(root)
        for page in doc.pages:
            self._build_page(root, page)
        root.setExpanded(True)

    def _build_page(self, parent: QTreeWidgetItem, page: PageNode) -> None:
        item = self._make_item(parent, "page", f"Page {page.page_number}", "", "", page)
        for block in page.blocks:
            self._build_block(item, block)
        if page.orphan_lines:
            grp = QTreeWidgetItem(item)
            grp.setText(0, f"~ Orphan lines ({len(page.orphan_lines)})")
            for line in page.orphan_lines:
                self._build_line(grp, line)

    def _build_block(self, parent: QTreeWidgetItem, block: BlockNode) -> None:
        label = f"[{block.label}]"
        item = self._make_item(parent, "block", label, block.content[:60], "", block)
        for line in block.lines:
            self._build_line(item, line)

    def _build_line(self, parent: QTreeWidgetItem, line: LineNode) -> None:
        conf_str = f"{line.confidence:.2f}" if line.confidence else ""
        item = self._make_item(parent, "line", "Line", line.text[:60], conf_str, line)
        if line.chars:
            ph = QTreeWidgetItem(item)
            ph.setText(0, f"  {len(line.chars)} chars (expand to load)")
            ph.setData(0, Qt.UserRole, ("char_placeholder", line))

    def _build_chars(self, parent: QTreeWidgetItem, line: LineNode) -> None:
        parent.takeChildren()
        for char in line.chars:
            conf_str = f"{char.confidence:.2f}"
            self._make_item(parent, "char", f"'{char.char}'", char.token_text, conf_str, char)

    def _make_item(self, parent, kind: str, label: str, text: str, conf: str, node: Any) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent) if parent else QTreeWidgetItem()
        item.setText(_COL_LABEL, label)
        item.setText(_COL_TEXT, text)
        item.setText(_COL_CONF, conf)
        item.setData(0, Qt.UserRole, node)
        item.setToolTip(_COL_TEXT, text)
        colours = {
            "document": QColor(200, 200, 255),
            "page":     QColor(180, 230, 255),
            "block":    QColor(160, 200, 255),
            "line":     QColor(200, 255, 200),
            "char":     QColor(255, 230, 180),
        }
        if kind in colours:
            item.setBackground(0, QBrush(colours[kind]))
        self._node_to_item[id(node)] = item
        return item

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "char_placeholder":
            self._build_chars(item, data[1])
            item.setExpanded(True)
            return
        if data is not None and not isinstance(data, tuple):
            if self._state.selected_node is not data:
                self._state.set_selection(data)

    def _on_external_selection(self, node: Any) -> None:
        if node is None:
            return
        item = self._node_to_item.get(id(node))
        if item is not None:
            self.scrollToItem(item, QAbstractItemView.PositionAtCenter)
            self.setCurrentItem(item)

    def expandItem(self, item: QTreeWidgetItem) -> None:
        if item.childCount() == 1:
            data = item.child(0).data(0, Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "char_placeholder":
                self._build_chars(item, data[1])
        super().expandItem(item)
