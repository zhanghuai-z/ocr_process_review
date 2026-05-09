"""Inspector panel — right pane showing attributes of selected IR node."""
from __future__ import annotations
from dataclasses import fields as dc_fields, is_dataclass
from typing import Any
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QLabel, QScrollArea, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)
from tools.ocr_inspector.state import AppState


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if is_dataclass(value) and not isinstance(value, type):
        lines = []
        for f in dc_fields(value):
            if f.name in ("raw", "chars", "lines", "blocks", "pages"):
                continue
            lines.append(f"  {f.name}: {_fmt(getattr(value, f.name))}")
        return "\n".join(lines) if lines else str(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            return f"[{', '.join(_fmt(v) for v in value)}]"
        return f"[{len(value)} items]"
    return str(value)


class InspectorPanel(QWidget):
    """Right pane: selected node attributes."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state

        self._title = QLabel("No selection")
        self._title.setFont(QFont("Consolas", 10, QFont.Bold))
        self._title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._title.setWordWrap(True)

        self._attrs_widget = QWidget()
        self._attrs_layout = QFormLayout(self._attrs_widget)
        self._attrs_layout.setLabelAlignment(Qt.AlignRight)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._attrs_widget)

        self._raw_view = QTextEdit()
        self._raw_view.setReadOnly(True)
        self._raw_view.setFont(QFont("Consolas", 8))
        self._raw_view.setMaximumHeight(200)
        self._raw_view.setPlaceholderText("raw JSON")

        raw_group = QGroupBox("Raw")
        rg_lay = QVBoxLayout(raw_group)
        rg_lay.addWidget(self._raw_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._title)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(raw_group)

        state.on_selection_changed(self._on_selection_changed)

    def _on_selection_changed(self, node: Any) -> None:
        self._clear_attrs()
        if node is None:
            self._title.setText("No selection")
            self._raw_view.clear()
            return

        self._title.setText(type(node).__name__)

        if is_dataclass(node) and not isinstance(node, type):
            for f in dc_fields(node):
                if f.name == "raw":
                    continue
                val = getattr(node, f.name)
                lbl = QLabel(f.name + ":")
                lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
                lbl.setFont(QFont("Consolas", 8))
                lbl.setMinimumWidth(100)

                display = _fmt(val)
                if "\n" in display:
                    vw = QTextEdit()
                    vw.setReadOnly(True)
                    vw.setFont(QFont("Consolas", 8))
                    vw.setPlainText(display)
                    vw.setMaximumHeight(80)
                    vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                else:
                    vw = QLabel(display)
                    vw.setFont(QFont("Consolas", 8))
                    vw.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    vw.setWordWrap(True)
                self._attrs_layout.addRow(lbl, vw)

        raw = getattr(node, "raw", None)
        if raw is not None:
            import json as _json
            try:
                self._raw_view.setPlainText(_json.dumps(raw, ensure_ascii=False, indent=2))
            except Exception:
                self._raw_view.setPlainText(str(raw))
        else:
            self._raw_view.clear()

    def _clear_attrs(self) -> None:
        while self._attrs_layout.rowCount() > 0:
            self._attrs_layout.removeRow(0)
