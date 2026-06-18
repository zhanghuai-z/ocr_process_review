"""内容区顶部 TopBar。

布局：
    项目 / 当前步骤          [版面分析] [横校] [纵校]          运行版面分析  导出

公开 API：
- 信号: ``step_clicked``, ``layout_run_clicked``, ``export_clicked``
- 方法: ``set_project_name(name)``, ``set_step_name(name)``,
        ``set_active(step)``, ``set_enabled_up_to(max_step)``,
        ``set_layout_run_enabled(bool)``, ``set_status(kind, text)``
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

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
        self.setFixedHeight(56)

        self._project_name: str = ""
        self._step_name: str = ""

        row = QHBoxLayout(self)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(12)

        # ── 面包屑 ─────────────────────────────
        left_box = QFrame()
        left_box.setObjectName("topBarLeft")
        left_row = QHBoxLayout(left_box)
        left_row.setContentsMargins(0, 0, 0, 0)
        left_row.setSpacing(8)

        self._project_lbl = QLabel("（无项目）")
        self._project_lbl.setObjectName("crumbProject")
        self._project_lbl.setMinimumWidth(160)
        self._project_lbl.setMaximumWidth(360)
        self._project_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        left_row.addWidget(self._project_lbl)

        self._sep2_lbl = QLabel("/")
        self._sep2_lbl.setObjectName("crumbSep")
        self._sep2_lbl.setVisible(False)
        left_row.addWidget(self._sep2_lbl)

        self._step_lbl = QLabel("")
        self._step_lbl.setObjectName("crumbStep")
        self._step_lbl.setVisible(False)
        left_row.addWidget(self._step_lbl)

        # ── 状态徽章 ─────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("statusPill")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setVisible(False)
        left_row.addSpacing(4)
        left_row.addWidget(self._status_lbl)
        left_row.addStretch(1)
        row.addWidget(left_box, 1)

        # ── 顶部流程入口 ─────────────────────────
        segment = QFrame()
        segment.setObjectName("workflowSegment")
        segment_row = QHBoxLayout(segment)
        segment_row.setContentsMargins(2, 2, 2, 2)
        segment_row.setSpacing(2)
        self._step_buttons: list[QPushButton] = []
        for label, target_step, _active in _WORKFLOW_STEPS:
            btn = QPushButton(label)
            btn.setObjectName("workflowStepBtn")
            btn.setCheckable(True)
            btn.setFixedWidth(104 if target_step == STEP_LAYOUT else 80)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _checked=False, s=target_step: self.step_clicked.emit(s))
            self._step_buttons.append(btn)
            segment_row.addWidget(btn)
        row.addWidget(segment, 0, Qt.AlignmentFlag.AlignCenter)

        # ── 右侧动作 ─────────────────────────────
        action_box = QFrame()
        action_box.setObjectName("topBarActions")
        action_row = QHBoxLayout(action_box)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(12)
        action_row.addStretch(1)

        self._btn_run_layout = QPushButton("▶ 运行版面分析")
        self._btn_run_layout.setObjectName("primaryBtn")
        self._btn_run_layout.setFixedWidth(138)
        self._btn_run_layout.setFixedHeight(32)
        self._btn_run_layout.setEnabled(False)
        self._btn_run_layout.clicked.connect(self.layout_run_clicked)
        action_row.addWidget(self._btn_run_layout)

        self._btn_export = QPushButton("⤓ 导出")
        self._btn_export.setObjectName("defaultBtn")
        self._btn_export.setFixedWidth(76)
        self._btn_export.setFixedHeight(32)
        self._btn_export.clicked.connect(self.export_clicked)
        action_row.addWidget(self._btn_export)
        row.addWidget(action_box, 1)

    # ── public API ──────────────────────────────────────────

    def set_layout_run_enabled(self, enabled: bool) -> None:
        self._btn_run_layout.setEnabled(enabled)

    def set_active(self, step: int) -> None:
        for i, (_, _, active_set) in enumerate(_WORKFLOW_STEPS):
            self._step_buttons[i].setChecked(step in active_set)

    def set_enabled_up_to(self, max_step: int) -> None:
        for i, (_, target_step, _) in enumerate(_WORKFLOW_STEPS):
            self._step_buttons[i].setEnabled(target_step <= max_step)

    def set_project_name(self, name: str) -> None:
        # 历史调用方传的是 "项目：xxx"，做一下兜底剥离
        if name.startswith("项目："):
            name = name[3:]
        self._project_name = (name or "").strip()
        self._refresh_crumb()

    def set_step_name(self, name: str) -> None:
        self._step_name = (name or "").strip()
        self._refresh_crumb()

    def set_status(self, kind: str, text: str = "") -> None:
        """更新状态徽章。kind ∈ {'idle','running','done','warn','hidden'}。"""
        kind = (kind or "").lower()
        if kind == "hidden" or not text:
            self._status_lbl.setVisible(False)
            return
        self._status_lbl.setText(text)
        # 用 dynamic property 让 QSS 命中 ``QLabel#statusPill[kind="done"]``
        self._status_lbl.setProperty("kind", kind)
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._status_lbl.setVisible(True)

    # ── 内部 ──────────────────────────────────────────────

    def _refresh_crumb(self) -> None:
        has_project = bool(self._project_name)
        has_step = bool(self._step_name)
        self._project_lbl.setText(self._project_name if has_project else "（无项目）")
        self._sep2_lbl.setVisible(has_step)
        self._step_lbl.setVisible(has_step)
        self._step_lbl.setText(self._step_name)


__all__ = ["TopBar"]
