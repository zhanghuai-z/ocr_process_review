"""正确率统计 对话框（原 h_proof 工具栏「评测：开/关」+「评测报告」入口）。

需求（Phase 11）：
- 不再用「掺沙子」入口名
- 入口移到「设置」菜单
- 名字改为 **正确率统计**
- 窗口至少包含：正确率统计开关 / 假象字详情 / 正确率计算

正确率公式（本轮模拟版）::

    rated = corrected + missed + edited_other
    if rated < min_observed_for_grade:
        rate = "样本不足"
    else:
        rate = corrected / rated  # 用户在被掺入假象字的位置识破并写回真字的比例

- ``deleted`` / ``pending`` 不计入分母
- 该比例越高，代表越能识破"形近假象字"
- 未来若引入"已校对总字数 / 实际错位"的真实正确率，会替换此公式

行为契约：
- 开 → ``qp.set_active_store(store)``；关 → ``qp.reset_active_store()``
- 任何路径**不会**改写 ``line.text``，导出文本永远不含假象字
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
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
    """正确率统计对话框。"""

    def __init__(
        self,
        project_provider: Callable[[], Optional[OcrProject]],
        refresh_panels_cb: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("正确率统计")
        self.setModal(True)
        self.resize(560, 520)
        self._project_provider = project_provider
        self._refresh_panels_cb = refresh_panels_cb
        self._build_ui()
        self._refresh_view()

    # ── UI ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # (1) 开关
        switch_box = QGroupBox("正确率统计开关")
        sl = QHBoxLayout(switch_box)
        self._btn_toggle = QPushButton("启用")
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.clicked.connect(self._on_toggle)
        self._switch_status = QLabel("当前：未启用")
        self._switch_status.setStyleSheet("color:#666;")
        sl.addWidget(self._btn_toggle)
        sl.addWidget(self._switch_status, 1)
        root.addWidget(switch_box)

        # (2) 假象字详情
        detail_box = QGroupBox("假象字详情")
        dl = QVBoxLayout(detail_box)
        self._detail_lbl = QLabel("尚未启用，无假象字。")
        self._detail_lbl.setStyleSheet("color:#666;")
        dl.addWidget(self._detail_lbl)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["页", "块", "行", "位置", "真字 → 假字 / 当前观察"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        dl.addWidget(self._table, 1)
        root.addWidget(detail_box, 1)

        # (3) 正确率计算
        rate_box = QGroupBox("正确率计算（模拟版，详见说明）")
        rl = QVBoxLayout(rate_box)
        self._rate_lbl = QLabel("rate = 待计算")
        self._rate_lbl.setStyleSheet("font-size:14px;")
        rl.addWidget(self._rate_lbl)
        formula = QLabel(
            "<small>"
            "<b>公式</b>：rate = corrected / (corrected + missed + edited_other)<br>"
            "<b>含义</b>：用户在被掺入假象字的位置识破并写回真字的比例。<br>"
            "<b>注</b>：<i>deleted</i> / <i>pending</i> 不计入分母；样本数过少时显示"
            " <i>样本不足</i>。<br>"
            "<b>边界</b>：本入口<u>不会</u>改写 line.text，导出文本永远不含假象字。"
            "</small>"
        )
        formula.setWordWrap(True)
        rl.addWidget(formula)
        root.addWidget(rate_box)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        root.addWidget(btn_box)

    # ── 行为 ──────────────────────────────────────────────────

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
            self._btn_toggle.setText("启用")
            self._switch_status.setText("当前：未启用")
            self._switch_status.setStyleSheet("color:#666;")
            self._detail_lbl.setText("尚未启用，无假象字。")
            self._table.setRowCount(0)
            self._rate_lbl.setText("rate = 待计算")
            return

        n = len(store)
        self._btn_toggle.setChecked(True)
        self._btn_toggle.setText("关闭")
        self._switch_status.setText(f"当前：已启用（{n} 处）")
        self._switch_status.setStyleSheet("color:#1a73e8;")
        self._detail_lbl.setText(f"共 {n} 个假象字位置：")

        probes = store.all()
        self._table.setRowCount(len(probes))
        for r, p in enumerate(probes):
            self._table.setItem(r, 0, QTableWidgetItem(str(p.key.page_number)))
            self._table.setItem(r, 1, QTableWidgetItem(str(p.key.block_index)))
            self._table.setItem(r, 2, QTableWidgetItem(str(p.key.line_index)))
            self._table.setItem(r, 3, QTableWidgetItem(str(p.key.char_index)))
            self._table.setItem(
                r, 4,
                QTableWidgetItem(
                    f"{p.true_char} \u2192 {p.fake_char}    [{p.observation}]"
                ),
            )

        rep = qp.score(store)
        rated = rep.corrected + rep.missed + rep.edited_other
        if rep.grade == "INSUFFICIENT" or rated == 0:
            self._rate_lbl.setText(
                f"rate = 样本不足  "
                f"（corrected={rep.corrected}, missed={rep.missed}, "
                f"edited_other={rep.edited_other}, deleted={rep.deleted}, "
                f"pending={rep.pending}）"
            )
        else:
            pct = rep.corrected / rated * 100
            self._rate_lbl.setText(
                f"rate = {pct:.1f}%  "
                f"（corrected={rep.corrected} / rated={rated}；等级={rep.grade_label}）"
            )
