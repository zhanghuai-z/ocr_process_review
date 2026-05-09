"""Inspector panel — right pane showing attributes of selected IR node.

Enhancements:
- source_field shown prominently at top with colour badge
- Coordinate space clarified: "原始图像像素空间"
- Element layer shown: Block / Line / Char
- bbox.area shown if bbox present
"""
from __future__ import annotations
from dataclasses import fields as dc_fields, is_dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QFormLayout, QFrame, QGroupBox, QLabel, QScrollArea, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

from tools.ocr_inspector.state import AppState
from tools.ocr_inspector.models.ir import BlockNode, CharNode, LineNode, BBox

# Source-field colour map (matches canvas colours)
_SOURCE_COLOURS: dict[str, str] = {
    "overall_ocr_res":         "#3C78FF",
    "overall_ocr_res.rec_texts": "#3C78FF",
    "parsing_res_list":        "#00C850",
    "layout_det_res":          "#A050FF",
    "text_word_region":        "#FFA000",
}
_FALLBACK_COLOUR = "#909090"


def _source_colour(sf: str) -> str:
    return _SOURCE_COLOURS.get(sf, _FALLBACK_COLOUR)


def _layer_name(node: Any) -> str:
    if isinstance(node, BlockNode):
        return "Block"
    if isinstance(node, LineNode):
        return "Line"
    if isinstance(node, CharNode):
        return "Char"
    return type(node).__name__


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


def _make_badge(text: str, colour: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Consolas", 8, QFont.Bold))
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background: {colour}; color: white; border-radius: 4px; padding: 2px 6px;"
    )
    lbl.setFixedHeight(22)
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class InspectorPanel(QWidget):
    """Right pane: selected node attributes."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state

        # ── Header area (title + source badge + layer badge) ───────────
        self._title = QLabel("No selection")
        self._title.setFont(QFont("Consolas", 10, QFont.Bold))
        self._title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._title.setWordWrap(True)

        self._badge_row = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        badge_layout = QHBoxLayout(self._badge_row)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(4)
        self._source_badge = QLabel()
        self._layer_badge  = QLabel()
        badge_layout.addWidget(self._source_badge)
        badge_layout.addWidget(self._layer_badge)
        badge_layout.addStretch()
        self._badge_row.setVisible(False)

        # Coord-space hint
        self._coord_hint = QLabel(
            "<small>坐标空间: <b>原始图像像素空间</b> — (x, y) 是图像左上角起的像素偏移</small>"
        )
        self._coord_hint.setWordWrap(True)
        self._coord_hint.setVisible(False)

        # ── Attribute form ─────────────────────────────────────────────
        self._attrs_widget = QWidget()
        self._attrs_layout = QFormLayout(self._attrs_widget)
        self._attrs_layout.setLabelAlignment(Qt.AlignRight)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._attrs_widget)

        # ── Raw JSON ──────────────────────────────────────────────────
        self._raw_view = QTextEdit()
        self._raw_view.setReadOnly(True)
        self._raw_view.setFont(QFont("Consolas", 8))
        self._raw_view.setMaximumHeight(200)
        self._raw_view.setPlaceholderText("raw JSON")

        raw_group = QGroupBox("Raw JSON (from adapter)")
        rg_lay = QVBoxLayout(raw_group)
        rg_lay.addWidget(self._raw_view)

        # ── Layout ────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._badge_row)
        layout.addWidget(self._coord_hint)
        layout.addWidget(_separator())
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(raw_group)

        state.on_selection_changed(self._on_selection_changed)

    # ─────────────────────────────────────────────────────────────────

    def _on_selection_changed(self, node: Any) -> None:
        self._clear_attrs()
        if node is None:
            self._title.setText("No selection")
            self._badge_row.setVisible(False)
            self._coord_hint.setVisible(False)
            self._raw_view.clear()
            return

        # Title: "Block · table"
        layer = _layer_name(node)
        kind_extra = getattr(node, "label", None) or getattr(node, "text", None) or getattr(node, "char", "")
        if kind_extra:
            self._title.setText(f"{layer}  ·  {str(kind_extra)[:60]}")
        else:
            self._title.setText(layer)

        # Source field badge
        sf = getattr(node, "source_field", "") or ""
        colour = _source_colour(sf)
        sf_text = sf if sf else "source unknown"
        self._source_badge.setText(sf_text)
        self._source_badge.setStyleSheet(
            f"background: {colour}; color: white; border-radius: 4px;"
            " padding: 2px 6px; font-family: Consolas; font-size: 8pt; font-weight: bold;"
        )
        self._source_badge.setFixedHeight(22)

        # Layer badge
        layer_colours = {"Block": "#336699", "Line": "#337766", "Char": "#996633"}
        lc = layer_colours.get(layer, "#666666")
        self._layer_badge.setText(layer)
        self._layer_badge.setStyleSheet(
            f"background: {lc}; color: white; border-radius: 4px;"
            " padding: 2px 6px; font-family: Consolas; font-size: 8pt;"
        )
        self._layer_badge.setFixedHeight(22)

        self._badge_row.setVisible(True)

        # Show coord hint if node has bbox
        has_bbox = getattr(node, "bbox", None) is not None
        self._coord_hint.setVisible(has_bbox)

        # Attributes
        if is_dataclass(node) and not isinstance(node, type):
            # Put source_field FIRST if it exists
            fnames = [f.name for f in dc_fields(node)]
            priority = ["id", "source_field", "label", "text", "char",
                        "confidence", "bbox", "polygon", "order", "content"]
            ordered = [n for n in priority if n in fnames]
            rest = [n for n in fnames if n not in priority and n != "raw"]
            for fname in ordered + rest:
                val = getattr(node, fname)
                # Extra info for bbox
                if fname == "bbox" and isinstance(val, BBox):
                    self._add_row(fname, _fmt(val))
                    self._add_row("  bbox.area (px²)", f"{val.area:.0f}")
                    continue
                self._add_row(fname, _fmt(val), highlight=(fname == "source_field"))

        raw = getattr(node, "raw", None)
        if raw is not None:
            import json as _json
            try:
                self._raw_view.setPlainText(_json.dumps(raw, ensure_ascii=False, indent=2))
            except Exception:
                self._raw_view.setPlainText(str(raw))
        else:
            self._raw_view.clear()

    def _add_row(self, name: str, value: str, highlight: bool = False) -> None:
        lbl = QLabel(name + ":")
        lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        lbl.setFont(QFont("Consolas", 8))
        lbl.setMinimumWidth(100)
        if highlight:
            lbl.setStyleSheet("color: #FFA040; font-weight: bold;")

        if "\n" in value:
            vw = QTextEdit()
            vw.setReadOnly(True)
            vw.setFont(QFont("Consolas", 8))
            vw.setPlainText(value)
            vw.setMaximumHeight(80)
            vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            vw = QLabel(value)
            vw.setFont(QFont("Consolas", 8))
            vw.setTextInteractionFlags(Qt.TextSelectableByMouse)
            vw.setWordWrap(True)
            if highlight:
                vw.setStyleSheet("color: #FFA040; font-weight: bold;")
        self._attrs_layout.addRow(lbl, vw)

    def _clear_attrs(self) -> None:
        while self._attrs_layout.rowCount() > 0:
            self._attrs_layout.removeRow(0)
