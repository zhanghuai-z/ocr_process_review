"""内容区顶部 TopBar（40 高）。

布局（对齐 Pencil 设计稿 xuwvF / SRPwN 顶栏）：
    OCR · 测试项目 / 版面分析    [已运行]              ▶ 运行  ⤓ 导出

公开 API：
- 信号: ``layout_run_clicked``, ``export_clicked``
- 方法: ``set_project_name(name)``, ``set_step_name(name)``,
        ``set_layout_run_enabled(bool)``, ``set_status(kind, text)``
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class TopBar(QWidget):
    layout_run_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(40)

        self._project_name: str = ""
        self._step_name: str = ""

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(8)

        # ── 面包屑（去掉左侧 OCR brand：意义重复且 NavRail 已表达） ──
        self._project_lbl = QLabel("（无项目）")
        self._project_lbl.setObjectName("crumbProject")
        row.addWidget(self._project_lbl)

        self._sep2_lbl = QLabel("/")
        self._sep2_lbl.setObjectName("crumbSep")
        self._sep2_lbl.setVisible(False)
        row.addWidget(self._sep2_lbl)

        self._step_lbl = QLabel("")
        self._step_lbl.setObjectName("crumbStep")
        self._step_lbl.setVisible(False)
        row.addWidget(self._step_lbl)

        # ── 状态徽章 ─────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("statusPill")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setVisible(False)
        row.addSpacing(8)
        row.addWidget(self._status_lbl)

        row.addStretch(1)

        # ── 右侧动作 ─────────────────────────────
        self._btn_run_layout = QPushButton("▶ 运行版面分析")
        self._btn_run_layout.setObjectName("primaryBtn")
        self._btn_run_layout.setEnabled(False)
        self._btn_run_layout.clicked.connect(self.layout_run_clicked)
        row.addWidget(self._btn_run_layout)

        self._btn_export = QPushButton("⤓ 导出")
        self._btn_export.setObjectName("ghostBtn")
        self._btn_export.clicked.connect(self.export_clicked)
        row.addWidget(self._btn_export)

    # ── public API ──────────────────────────────────────────

    def set_layout_run_enabled(self, enabled: bool) -> None:
        self._btn_run_layout.setEnabled(enabled)

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
