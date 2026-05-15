"""正确率统计 对话框（Phase 24 纯功能型重构）。

用户要求（Phase 24）：
- 去掉一大段说明型文案
- 入口为"开始统计"按钮
- 主体是一张表，每个假象字位置一行
- 列出真字 / 假字 / 用户当前观察 / **是否修正**
- 底部直接显示"正确率 X.X%"

行为契约（保持与 Phase 11 一致）：
- 开 → ``qp.set_active_store(store)``；关 → ``qp.reset_active_store()``
- 任何路径**不会**改写 ``line.text``，导出文本永远不含假象字

测试兼容性（沿用 Phase 11+）：
- 仍保留 ``_btn_toggle`` / ``_switch_status`` / ``_table`` / ``_rate_lbl``
  / ``_on_toggle`` 字段名，避免外部测试断裂。
- ``_switch_status`` 文案前缀 ``当前：未启用`` / ``当前：已启用`` 不变。
- ``_rate_lbl.text()`` 初始值 ``rate = 待计算`` 不变。
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.core import quality_probe as qp
from app.models import OcrProject


class QualityStatsDialog(QDialog):
    """正确率统计对话框（Phase 24：纯功能型）。"""

    def __init__(
        self,
        project_provider: Callable[[], Optional[OcrProject]],
        refresh_panels_cb: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("正确率统计")
        self.setModal(True)
        self.resize(640, 480)
        self._project_provider = project_provider
        self._refresh_panels_cb = refresh_panels_cb
        self._build_ui()
        self._refresh_view()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # 顶部：开始/停止统计 + 状态
        top = QHBoxLayout()
        top.setSpacing(8)
        self._btn_toggle = QPushButton("开始统计")
        self._btn_toggle.setObjectName("primaryBtn")
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.setMinimumHeight(30)
        self._btn_toggle.clicked.connect(self._on_toggle)
        top.addWidget(self._btn_toggle)
        self._switch_status = QLabel("当前：未启用")
        self._switch_status.setStyleSheet("color:#666;")
        top.addWidget(self._switch_status, 1)
        root.addLayout(top)

        # 中部：详情表格（一字一行）
        # 列：页 / 块 / 行 / 位置 / 真字 / 假字 / 是否修正 / 当前观察
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["页", "块", "行", "位置", "真字", "假字", "是否修正", "当前观察"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self._table, 1)

        # 底部：正确率 + 关闭
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._rate_lbl = QLabel("rate = 待计算")
        self._rate_lbl.setStyleSheet(
            "font-size:15px; font-weight:600; color:#1a73e8;"
        )
        bottom.addWidget(self._rate_lbl, 1)
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        bottom.addWidget(btn_box)
        root.addLayout(bottom)

    # ── 行为 ───────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        if checked:
            project = self._project_provider()
            if project is None or not project.pages:
                QMessageBox.information(
                    self, "正确率统计",
                    "尚无校对数据，请先完成 OCR。",
                )
                self._btn_toggle.setChecked(False)
                return
            cfg = qp.sampler_config_from_app_config()
            store = qp.ProbeSampler(cfg).sample(project)
            qp.set_active_store(store)
            if len(store) == 0:
                QMessageBox.information(
                    self, "正确率统计",
                    "当前正文中没有可投放假象字的位置（页面太短或全是数字/公式/标题）。",
                )
                qp.reset_active_store()
                self._btn_toggle.setChecked(False)
                return
        else:
            qp.reset_active_store()
        try:
            self._refresh_panels_cb()
        except Exception:
            pass
        self._refresh_view()

    def _refresh_view(self) -> None:
        store = qp.get_active_store()
        if store is None or len(store) == 0:
            self._btn_toggle.setChecked(False)
            self._btn_toggle.setText("开始统计")
            self._switch_status.setText("当前：未启用")
            self._switch_status.setStyleSheet("color:#666;")
            self._table.setRowCount(0)
            self._rate_lbl.setText("rate = 待计算")
            self._rate_lbl.setStyleSheet(
                "font-size:15px; font-weight:600; color:#999;"
            )
            return

        n = len(store)
        self._btn_toggle.setChecked(True)
        self._btn_toggle.setText("停止统计")
        self._switch_status.setText(f"当前：已启用（{n} 处）")
        self._switch_status.setStyleSheet("color:#1a73e8;")

        probes = store.all()
        self._table.setRowCount(len(probes))
        for r, p in enumerate(probes):
            obs = str(p.observation)
            # observation == "corrected" 视为已修正；其它（missed/edited_other/
            # deleted/pending）一律视为未修正。
            corrected = obs.lower().endswith("corrected") or obs == "corrected"
            corrected_item = QTableWidgetItem("✓ 已修正" if corrected else "✗ 未修正")
            corrected_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            corrected_item.setForeground(
                Qt.GlobalColor.darkGreen if corrected else Qt.GlobalColor.darkRed
            )
            self._table.setItem(r, 0, QTableWidgetItem(str(p.key.page_number)))
            self._table.setItem(r, 1, QTableWidgetItem(str(p.key.block_index)))
            self._table.setItem(r, 2, QTableWidgetItem(str(p.key.line_index)))
            self._table.setItem(r, 3, QTableWidgetItem(str(p.key.char_index)))
            self._table.setItem(r, 4, QTableWidgetItem(p.true_char))
            self._table.setItem(r, 5, QTableWidgetItem(p.fake_char))
            self._table.setItem(r, 6, corrected_item)
            self._table.setItem(r, 7, QTableWidgetItem(obs))

        rep = qp.score(store)
        rated = rep.corrected + rep.missed + rep.edited_other
        if rep.grade == "INSUFFICIENT" or rated == 0:
            self._rate_lbl.setText(
                f"正确率：样本不足 "
                f"（已修正 {rep.corrected} / 总观察 {rated}）"
            )
            self._rate_lbl.setStyleSheet(
                "font-size:15px; font-weight:600; color:#999;"
            )
        else:
            pct = rep.corrected / rated * 100
            self._rate_lbl.setText(
                f"正确率：{pct:.1f}%   "
                f"（已修正 {rep.corrected} / 总观察 {rated}）"
            )
            self._rate_lbl.setStyleSheet(
                "font-size:15px; font-weight:600; color:#1a73e8;"
            )
