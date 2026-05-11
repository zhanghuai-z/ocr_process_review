"""Parse Log panel — shows messages recorded during adapter parsing.

Subscribes to state.on_document_changed().
Reads doc.parse_log (list of ParseLogEntry).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from tools.ocr_inspector.state import AppState

_LEVEL_COLOURS = {
    "ERROR":   QColor(255,  80,  80),
    "WARNING": QColor(255, 200,  50),
    "INFO":    QColor(120, 200, 120),
    "DEBUG":   QColor(160, 160, 160),
}


class ParseLogPanel(QWidget):
    """Displays parse_log entries from the current DocumentNode."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>解析日志 (Parse Log)</b>"))
        hdr.addStretch()
        self._count_label = QLabel("0 条")
        hdr.addWidget(self._count_label)
        btn_clear = QPushButton("清空")
        btn_clear.setFixedWidth(50)
        btn_clear.clicked.connect(self._clear)
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["级别", "阶段/来源", "消息"])
        self._tree.setColumnWidth(0, 70)
        self._tree.setColumnWidth(1, 140)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setWordWrap(True)
        layout.addWidget(self._tree)

        state.on_document_changed(self._on_doc_changed)

    # ── state callback ────────────────────────────────────────────────────

    def _on_doc_changed(self, doc) -> None:
        self._tree.clear()
        if doc is None:
            self._count_label.setText("0 条")
            return
        entries = getattr(doc, "parse_log", []) or []
        for entry in entries:
            self._add_entry(entry)
        self._count_label.setText(f"{len(entries)} 条")

    def _add_entry(self, entry) -> None:
        """Accept dict, dataclass, or plain str entries from adapter."""
        if isinstance(entry, str):
            # Plain string: parse optional "LEVEL: message" prefix
            for lvl in ("ERROR", "WARNING", "INFO", "DEBUG"):
                if entry.upper().startswith(lvl + ":") or entry.upper().startswith(lvl + " "):
                    level = lvl
                    msg   = entry[len(lvl)+1:].lstrip(": ").strip()
                    stage = ""
                    break
            else:
                level = "INFO"
                stage = ""
                msg   = entry
        elif isinstance(entry, dict):
            level = str(entry.get("level", "INFO")).upper()
            stage = str(entry.get("stage", ""))
            msg   = str(entry.get("message", ""))
        else:
            level = str(getattr(entry, "level", "INFO")).upper()
            stage = str(getattr(entry, "stage", ""))
            msg   = str(getattr(entry, "message", ""))

        item = QTreeWidgetItem(self._tree, [level, stage, msg])
        colour = _LEVEL_COLOURS.get(level, QColor(200, 200, 200))
        item.setForeground(0, QBrush(colour))
        item.setToolTip(2, msg)
        self._tree.addTopLevelItem(item)

    def _clear(self) -> None:
        self._tree.clear()
        self._count_label.setText("0 条")
