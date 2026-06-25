"""Small visual helpers shared by Qt widgets."""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def apply_soft_shadow(
    widget: QWidget,
    *,
    blur_radius: float = 20.0,
    x_offset: float = 0.0,
    y_offset: float = 4.0,
    alpha: int = 18,
) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur_radius)
    effect.setOffset(x_offset, y_offset)
    effect.setColor(QColor(0, 0, 0, max(0, min(alpha, 255))))
    widget.setGraphicsEffect(effect)
