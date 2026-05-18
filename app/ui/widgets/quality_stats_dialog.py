"""正确率统计 对话框（proof UI clarity 重构）。

设计变化（相对 Phase 24）：
- 主窗变小：开关 + 圆环正确率 + 详情入口（按钮），不再把大表格堆在主窗
- 假象字明细放进 ``QualityStatsDetailDialog``，由"详情…"按钮打开
- 圆环 ``_RatioRing`` 自绘，避免引入 QtCharts 依赖

行为契约（保持与 Phase 11+ / Phase 24 一致）：
- 开 → ``qp.set_active_store(store)``；关 → ``qp.reset_active_store()``
- 任何路径**不会**改写 ``line.text``，导出文本永远不含假象字

测试兼容性：
- 保留 ``_btn_toggle`` / ``_switch_status`` / ``_table`` / ``_rate_lbl`` /
  ``_on_toggle`` 字段名。``_table`` 现在指向详情子窗内的 QTableWidget（懒创建）。
- ``_switch_status`` 文案前缀 ``当前：未启用`` / ``当前：已启用`` 不变。
- ``_rate_lbl.text()`` 初始值 ``rate = 待计算`` 不变（控件 hidden，仅供测试断言）。
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core import quality_probe as qp
from app.models import OcrProject


# ─────────────────────────────────────────────────────────────
# 自绘圆环正确率
# ─────────────────────────────────────────────────────────────

class _RatioRing(QWidget):
    """正确率圆环：外环按 ratio 上色，中心显示百分比 + 副标签。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ratio: Optional[float] = None  # None = 未启用 / 样本不足
        self._sub_text: str = ""
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_ratio(self, ratio: Optional[float], sub_text: str = "") -> None:
        self._ratio = ratio
        self._sub_text = sub_text
        self.update()

    def paintEvent(self, _ev) -> None:  # type: ignore[override]
        size = min(self.width(), self.height())
        margin = 8
        rect = QRectF(
            (self.width() - size) / 2 + margin,
            (self.height() - size) / 2 + margin,
            size - 2 * margin,
            size - 2 * margin,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 底环（灰）
        pen = QPen(QColor("#e3e8ef"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        # 数据环 + 中心文本
        if self._ratio is None:
            color = QColor("#9aa4b2")
            center_main = "—"
        else:
            r = max(0.0, min(1.0, self._ratio))
            if r >= 0.95:
                color = QColor("#2e7d32")    # 绿
            elif r >= 0.80:
                color = QColor("#1a73e8")    # 蓝
            elif r >= 0.60:
                color = QColor("#f9a825")    # 黄
            else:
                color = QColor("#c62828")    # 红
            sweep = int(round(r * 360 * 16))
            pen = QPen(color, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            # 从 12 点钟方向起，顺时针绘制 → Qt 角度系统中起点 = 90*16，
            # sweep 为负即顺时针。
            painter.drawArc(rect, 90 * 16, -sweep)
            # Task #4: 保留两位小数
            center_main = f"{r * 100:.2f}%"

        # 中心主数字
        painter.setPen(QColor("#222"))
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        painter.setFont(f)
        text_rect = rect.adjusted(0, 0, 0, -int(rect.height() * 0.28))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, center_main)

        # 副文本（已修正/总观察）
        if self._sub_text:
            painter.setPen(QColor("#666"))
            f2 = QFont()
            f2.setPointSize(9)
            painter.setFont(f2)
            sub_rect = rect.adjusted(0, int(rect.height() * 0.32), 0, 0)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, self._sub_text)

        painter.end()


# ─────────────────────────────────────────────────────────────
# 详情子窗：假象字明细表格
# ─────────────────────────────────────────────────────────────

class QualityStatsDetailDialog(QDialog):
    """假象字位置明细子窗（从主窗 '详情…' 入口打开）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("正确率统计 · 假象字明细")
        self.setModal(True)
        self.resize(720, 460)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self._summary = QLabel("尚未启用")
        self._summary.setStyleSheet("color:#666;")
        root.addWidget(self._summary)

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

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        root.addWidget(btn_box)

    def populate(self) -> None:
        """从全局 active store 拉取最新明细并填表。"""
        store = qp.get_active_store()
        if store is None or len(store) == 0:
            self._table.setRowCount(0)
            self._summary.setText("尚未启用")
            return
        probes = store.all()
        self._summary.setText(f"共 {len(probes)} 个假象字位置")
        self._table.setRowCount(len(probes))
        for r, p in enumerate(probes):
            obs = str(p.observation)
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


# ─────────────────────────────────────────────────────────────
# 主窗：开关 + 圆环 + 详情入口
# ─────────────────────────────────────────────────────────────

class QualityStatsDialog(QDialog):
    """正确率统计主窗（proof UI clarity：紧凑型）。"""

    def __init__(
        self,
        project_provider: Callable[[], Optional[OcrProject]],
        refresh_panels_cb: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("正确率统计")
        self.setModal(True)
        # 紧凑主窗：360 × 320 足够开关 + 圆环 + 详情入口
        self.resize(360, 320)
        self._project_provider = project_provider
        self._refresh_panels_cb = refresh_panels_cb
        self._detail_dialog: Optional[QualityStatsDetailDialog] = None
        self._build_ui()
        self._refresh_view()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # 顶部：开关 + 状态
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

        # 中部：圆环
        self._ring = _RatioRing()
        root.addWidget(self._ring, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        # 详情入口 + 关闭
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._btn_detail = QPushButton("详情…")
        self._btn_detail.setMinimumHeight(28)
        self._btn_detail.setToolTip("查看每个假象字位置的修正情况")
        self._btn_detail.clicked.connect(self._open_detail)
        bottom.addWidget(self._btn_detail)
        bottom.addStretch()
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        bottom.addWidget(btn_box)
        root.addLayout(bottom)

        # 隐藏的兼容字段：保留旧测试断言用的 _rate_lbl 文本接口
        self._rate_lbl = QLabel("rate = 待计算", self)
        self._rate_lbl.hide()

    # ── 兼容外部测试：暴露详情表格 ──────────────────────────────

    @property
    def _table(self) -> QTableWidget:
        """旧测试访问主窗 _table；现在指向详情子窗的表格（懒创建）。"""
        return self._ensure_detail_dialog()._table

    def _ensure_detail_dialog(self) -> QualityStatsDetailDialog:
        if self._detail_dialog is None:
            self._detail_dialog = QualityStatsDetailDialog(self)
        return self._detail_dialog

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

    def _open_detail(self) -> None:
        dlg = self._ensure_detail_dialog()
        dlg.populate()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _refresh_view(self) -> None:
        store = qp.get_active_store()
        if store is None or len(store) == 0:
            self._btn_toggle.setChecked(False)
            self._btn_toggle.setText("开始统计")
            self._switch_status.setText("当前：未启用")
            self._switch_status.setStyleSheet("color:#666;")
            self._ring.set_ratio(None, "")
            self._rate_lbl.setText("rate = 待计算")
            self._ensure_detail_dialog().populate()
            return

        n = len(store)
        self._btn_toggle.setChecked(True)
        self._btn_toggle.setText("停止统计")
        self._switch_status.setText(f"当前：已启用（{n} 处）")
        self._switch_status.setStyleSheet("color:#1a73e8;")

        rep = qp.score(store)
        rated = rep.corrected + rep.missed + rep.edited_other
        # Task #4：不再展示"样本不足"文案；rated==0 时按 0.00% 处理。
        # rep.grade 仍可能是 INSUFFICIENT（quality_probe 核心语义保留），
        # 但 UI 层一律以 ratio=corrected/max(rated,1) 显示。
        if rated == 0:
            ratio = 0.0
        else:
            ratio = rep.corrected / rated
        self._ring.set_ratio(ratio, f"已修正 {rep.corrected} / 总观察 {rated}")
        self._rate_lbl.setText(
            f"rate = {ratio * 100:.2f}% "
            f"(corrected={rep.corrected} / rated={rated})"
        )

        # 始终把最新明细同步到详情子窗（隐藏即可，便于测试/外部直接读 _table）
        self._ensure_detail_dialog().populate()
