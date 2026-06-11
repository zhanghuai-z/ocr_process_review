"""左侧竖向 NavRail（56 宽）。

- 顶部：3 个步骤图标按钮（版面 / 横校 / 纵校），垂直排列，互斥 checked
- 底部：设置齿轮（仅此一个）

「▶ 运行版面分析」「⤓ 导出」「项目名」迁移到 TopBar（见 top_bar.py），不在本控件。
"""
from __future__ import annotations
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.controllers.workflow_controller import (
    STEP_LAYOUT, STEP_HPROOF, STEP_VPROOF,
)


# (tooltip, glyph, target_step, active_on_steps)
# 单字图标确保跨平台渲染、语义直白
_NAV_ITEMS = [
    ("版面分析", "版", STEP_LAYOUT,  frozenset({STEP_LAYOUT})),
    ("横向校对", "横", STEP_HPROOF, frozenset({STEP_HPROOF})),
    ("纵向校对", "纵", STEP_VPROOF, frozenset({STEP_VPROOF})),
]


class _NavIconButton(QPushButton):
    """NavRail 内的图标按钮：44×44，单字图标。"""
    def __init__(self, glyph: str, tooltip: str, parent=None):
        super().__init__(glyph, parent)
        self.setObjectName("navIcon")
        self.setToolTip(tooltip)
        self.setFixedSize(44, 44)
        self.setCheckable(True)
        # 加大字号让单字图标更显眼
        f = self.font()
        f.setPointSize(max(f.pointSize() + 4, 14))
        f.setBold(True)
        self.setFont(f)


class NavRail(QWidget):
    """左侧竖向导航栏。"""
    step_clicked = Signal(int)
    settings_clicked = Signal()

    WIDTH = 56

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self.setFixedWidth(self.WIDTH)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 16, 0, 16)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # 步骤按钮
        self._buttons: List[_NavIconButton] = []
        for tooltip, glyph, target_step, _active in _NAV_ITEMS:
            btn = _NavIconButton(glyph, tooltip)
            btn.clicked.connect(lambda _, s=target_step: self.step_clicked.emit(s))
            self._buttons.append(btn)
            root.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addStretch(1)

        # 底部：仅设置按钮（放大到 44，并加大字号）
        self._btn_settings = QPushButton("⚙")
        self._btn_settings.setObjectName("navIcon")
        self._btn_settings.setToolTip("设置")
        self._btn_settings.setFixedSize(44, 44)
        self._btn_settings.clicked.connect(self.settings_clicked)
        # 略加大设置图标字号（QSS 已规定基础 size，这里用样式覆盖）
        f = self._btn_settings.font()
        f.setPointSize(max(f.pointSize() + 4, 14))
        self._btn_settings.setFont(f)
        root.addWidget(self._btn_settings, alignment=Qt.AlignmentFlag.AlignHCenter)

    # ── 与旧 TopNavBar 接口对齐 ──────────────────────────────

    def set_active(self, step: int) -> None:
        for i, (_, _, _, active_set) in enumerate(_NAV_ITEMS):
            self._buttons[i].setChecked(step in active_set)

    def set_enabled_up_to(self, max_step: int) -> None:
        for i, (_, _, target_step, _) in enumerate(_NAV_ITEMS):
            self._buttons[i].setEnabled(target_step <= max_step)


__all__ = ["NavRail"]
