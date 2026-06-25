"""内容区顶部 TopBar。

布局：
    项目名                       [版面分析] [横校] [纵校]      文件 导出 更多 / 运行版面分析

公开 API：
- 信号: ``step_clicked``, ``layout_run_clicked``, ``export_clicked``
- 方法: ``set_project_name(name)``, ``set_active(step)``,
        ``set_enabled_up_to(max_step)``, ``set_layout_run_enabled(bool)``,
        ``set_menus(file_menu, more_menu)``
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QFrame, QMenu,
)
from app.ui.widgets.effects import apply_soft_shadow
from app.utils.icon_manager import get_icon

from app.controllers.workflow_controller import STEP_HPROOF, STEP_LAYOUT, STEP_OCR, STEP_VPROOF


_WORKFLOW_STEPS = [
    ("版面分析", STEP_LAYOUT, frozenset({STEP_LAYOUT, STEP_OCR})),
    ("横校", STEP_HPROOF, frozenset({STEP_HPROOF})),
    ("纵校", STEP_VPROOF, frozenset({STEP_VPROOF})),
]


class TopBar(QWidget):
    step_clicked = Signal(int)
    layout_run_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(58)

        self._project_name: str = ""

        row = QHBoxLayout(self)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(14)

        # ── Project identity ───────────────────────────────────
        left_box = QFrame()
        left_box.setObjectName("topBarLeft")
        left_row = QHBoxLayout(left_box)
        left_row.setContentsMargins(0, 0, 0, 0)
        left_row.setSpacing(12)

        self._title_lbl = QLabel("未命名项目")
        self._title_lbl.setObjectName("brandTitle")
        left_row.addWidget(self._title_lbl)
        left_row.addStretch(1)
        row.addWidget(left_box, 2)

        # ── 顶部流程入口 ─────────────────────────
        segment = QFrame()
        segment.setObjectName("pillSwitch")
        apply_soft_shadow(segment, blur_radius=18, y_offset=4, alpha=16)
        segment_row = QHBoxLayout(segment)
        segment_row.setContentsMargins(4, 4, 4, 4)
        segment_row.setSpacing(4)
        self._step_buttons: list[QPushButton] = []
        for label, target_step, _active in _WORKFLOW_STEPS:
            btn = QPushButton(label)
            btn.setObjectName("workflowStepBtn")
            btn.setCheckable(True)
            btn.setFixedWidth(104 if label == "版面分析" else 88)
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, s=target_step: self.step_clicked.emit(s))
            self._step_buttons.append(btn)
            segment_row.addWidget(btn)
        row.addWidget(segment, 0, Qt.AlignmentFlag.AlignCenter)

        # ── 右侧动作 ─────────────────────────────
        action_box = QFrame()
        action_box.setObjectName("topBarActions")
        action_row = QHBoxLayout(action_box)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addStretch(1)

        self._btn_file_menu = QPushButton("文件")
        self._btn_file_menu.setIcon(get_icon("folder", color="#6B6B6B"))
        self._btn_file_menu.setIconSize(QSize(16, 16))
        self._btn_file_menu.setObjectName("topMenuBtn")
        self._btn_file_menu.setFixedHeight(30)
        self._btn_file_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        action_row.addWidget(self._btn_file_menu)

        self._btn_export = QPushButton("导出")
        self._btn_export.setIcon(get_icon("export", color="#6B6B6B"))
        self._btn_export.setIconSize(QSize(16, 16))
        self._btn_export.setObjectName("topMenuBtn")
        self._btn_export.setFixedHeight(30)
        self._btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_export.clicked.connect(self.export_clicked)
        action_row.addWidget(self._btn_export)

        self._btn_more_menu = QPushButton("更多")
        self._btn_more_menu.setIcon(get_icon("menu", color="#6B6B6B"))
        self._btn_more_menu.setIconSize(QSize(16, 16))
        self._btn_more_menu.setObjectName("topMenuBtn")
        self._btn_more_menu.setFixedHeight(30)
        self._btn_more_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        action_row.addWidget(self._btn_more_menu)

        self._btn_run_layout = QPushButton("运行版面分析")
        self._btn_run_layout.setIcon(get_icon("play", color="#FFFFFF", stroke_width=2.0))
        self._btn_run_layout.setIconSize(QSize(15, 15))
        self._btn_run_layout.setObjectName("darkBtn")
        self._btn_run_layout.setFixedHeight(30)
        self._btn_run_layout.setMinimumWidth(136)
        self._btn_run_layout.setEnabled(False)
        self._btn_run_layout.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_run_layout.clicked.connect(self.layout_run_clicked)
        apply_soft_shadow(self._btn_run_layout, blur_radius=16, y_offset=3, alpha=18)
        action_row.addWidget(self._btn_run_layout)
        row.addWidget(action_box, 2) # Action box also stretches, keeping segment centered

    # ── public API ──────────────────────────────────────────

    def set_layout_run_enabled(self, enabled: bool) -> None:
        self._btn_run_layout.setEnabled(enabled)

    def set_active(self, step: int) -> None:
        for i, (_, _, active_set) in enumerate(_WORKFLOW_STEPS):
            self._step_buttons[i].setChecked(step in active_set)
        self._btn_run_layout.setVisible(step in (STEP_LAYOUT, STEP_OCR))

    def set_enabled_up_to(self, max_step: int) -> None:
        for i, (_, target_step, _) in enumerate(_WORKFLOW_STEPS):
            self._step_buttons[i].setEnabled(target_step <= max_step)

    def set_menus(
        self,
        file_menu: QMenu | None,
        more_menu: QMenu | None,
    ) -> None:
        self._btn_file_menu.setMenu(file_menu)
        self._btn_more_menu.setMenu(more_menu)

    def set_project_name(self, name: str) -> None:
        self._project_name = (name or "").strip()
        self._refresh_crumb()

    # ── 内部 ──────────────────────────────────────────────

    def _refresh_crumb(self) -> None:
        self._title_lbl.setText(self._project_name or "未命名项目")


__all__ = ["TopBar"]
