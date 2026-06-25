"""Semantic vector icons for the Qt UI."""
from __future__ import annotations

from html import escape

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


_ICON_BODY: dict[str, str] = {
    "folder": '<path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M3 9h18"/>',
    "save": '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/>',
    "export": '<path d="M12 4v10"/><path d="m8 8 4-4 4 4"/><path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>',
    "undo": '<path d="M9 7H4v5"/><path d="M4 12a8 8 0 1 0 2.35-5.65L4 8.7"/>',
    "redo": '<path d="M15 7h5v5"/><path d="M20 12a8 8 0 1 1-2.35-5.65L20 8.7"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.6-3.6"/>',
    "delete": '<path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 14h10l1-14"/><path d="M9 7V4h6v3"/>',
    "doc_image": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5"/><circle cx="10" cy="13" r="1.5"/><path d="m8 18 3-3 2 2 2-2 2 3"/>',
    "layout_box": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16"/><path d="M10 10v10"/><path d="M14 10v4h6"/>',
    "text": '<path d="M5 5h14"/><path d="M12 5v14"/><path d="M9 19h6"/>',
    "formula": '<path d="M17 5H8l5 7-5 7h9"/><path d="M15 9h4"/><path d="M15 15h4"/>',
    "table": '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 10h16"/><path d="M4 15h16"/><path d="M10 5v14"/><path d="M15 5v14"/>',
    "picture": '<rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="m6 17 4-4 3 3 2-2 3 3"/>',
    "heading": '<path d="M5 5v14"/><path d="M19 5v14"/><path d="M5 12h14"/><path d="M14 19h6"/>',
    "zoom": '<circle cx="11" cy="11" r="7"/><path d="M11 8v6"/><path d="M8 11h6"/><path d="m20 20-3.6-3.6"/>',
    "hand": '<path d="M8 12V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 11V4a1.5 1.5 0 0 1 3 0v7"/><path d="M14 11V6a1.5 1.5 0 0 1 3 0v7"/><path d="M17 13v-2a1.5 1.5 0 0 1 3 0v4c0 4-2.5 6-6 6h-2c-2.5 0-4.2-1.2-5.5-3.2L4 14.5a1.7 1.7 0 0 1 2.6-2.1L8 14"/>',
    "play": '<path d="M8 5v14l11-7z"/>',
    "settings": '<path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z"/><path d="M4 13.5v-3l2.2-.5.6-1.5L5.6 6.6l2-2 1.9 1.2 1.5-.6.5-2.2h3l.5 2.2 1.5.6 1.9-1.2 2 2-1.2 1.9.6 1.5 2.2.5v3l-2.2.5-.6 1.5 1.2 1.9-2 2-1.9-1.2-1.5.6-.5 2.2h-3l-.5-2.2-1.5-.6-1.9 1.2-2-2 1.2-1.9-.6-1.5z"/>',
    "api": '<path d="M7 8 3 12l4 4"/><path d="m17 8 4 4-4 4"/><path d="m14 5-4 14"/>',
    "segment": '<path d="M4 7h16"/><path d="M4 12h10"/><path d="M4 17h7"/><path d="M17 11l3 3-3 3"/>',
    "spellcheck": '<path d="m4 17 4-10 4 10"/><path d="M6 13h4"/><path d="m14 15 2 2 4-5"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
    "keyboard": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M19 10h.01M7 14h.01M11 14h6"/>',
    "cache": '<path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.6 9A7 7 0 0 1 18.5 7.5L20 12"/><path d="M18.4 15A7 7 0 0 1 5.5 16.5L4 12"/>',
    "menu": '<path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/>',
    "sidebar": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 4v16"/><path d="m14 9 3 3-3 3"/>',
    "directory": '<path d="M7 6h14"/><path d="M7 12h14"/><path d="M7 18h14"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
}


_ALIASES = {
    "charocr": "spellcheck",
    "shortcuts": "keyboard",
}


class IconManager:
    """Render and cache small line icons."""

    _cache: dict[tuple[str, str, int, float], QIcon] = {}

    @classmethod
    def get_icon(
        cls,
        name: str,
        *,
        color: str = "#6B6B6B",
        size: int = 24,
        stroke_width: float = 1.8,
    ) -> QIcon:
        resolved = _ALIASES.get(name, name)
        body = _ICON_BODY.get(resolved)
        if not body:
            return QIcon()
        key = (resolved, color, size, stroke_width)
        if key in cls._cache:
            return cls._cache[key]

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            'viewBox="0 0 24 24" fill="none" stroke="{color}" '
            'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            "{body}</svg>"
        ).format(
            size=size,
            color=escape(color, quote=True),
            stroke_width=stroke_width,
            body=body,
        )
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        icon = QIcon(pixmap)
        cls._cache[key] = icon
        return icon


def get_icon(
    name: str,
    *,
    color: str = "#6B6B6B",
    size: int = 24,
    stroke_width: float = 1.8,
) -> QIcon:
    """Return a semantic vector icon."""

    return IconManager.get_icon(
        name,
        color=color,
        size=size,
        stroke_width=stroke_width,
    )
