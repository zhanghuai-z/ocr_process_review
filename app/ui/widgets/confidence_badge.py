"""置信度徽章：小标签，颜色根据置信度变化。"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel


def normalize_badge_score(score: float) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    return max(0.0, min(value, 1.0))


def confidence_color(score: float) -> str:
    """返回 CSS 颜色字符串。"""
    score = normalize_badge_score(score)
    if score >= 0.90:
        return "#4CAF50"   # 绿
    elif score >= 0.75:
        return "#FF9800"   # 橙
    else:
        return "#F44336"   # 红


class ConfidenceBadge(QLabel):
    """显示置信度百分比的小徽章标签。"""

    def __init__(self, score: float = 1.0, parent=None):
        super().__init__(parent)
        self.set_score(score)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(52)

    def set_score(self, score: float) -> None:
        self._score = normalize_badge_score(score)
        color = confidence_color(self._score)
        pct = int(self._score * 100)
        self.setText(f"{pct}%")
        self.setStyleSheet(
            f"background:{color}; color:#fff; border-radius:4px;"
            f"padding:1px 5px; font-size:11px; font-weight:bold;"
        )
        self.setToolTip(f"置信度: {self._score:.4f}")
