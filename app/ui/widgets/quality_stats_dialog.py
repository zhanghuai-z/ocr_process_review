"""抽样字符校对观察 对话框（proof UI clarity 重构）。

设计变化（相对 Phase 24）：
- 主窗变小：开关 + 抽样观察圆环 + 详情入口（按钮），不再把大表格堆在主窗
- 假象字明细放进 ``QualityStatsDetailDialog``，由"详情…"按钮打开
- 圆环 ``_RatioRing`` 自绘，避免引入 QtCharts 依赖

行为契约（保持与 Phase 11+ / Phase 24 一致）：
- 开 → ``qp.set_active_store(store)``；关 → ``qp.reset_active_store()``
- 任何路径**不会**把假象字写入 ``line.final_text``，导出文本永远不含假象字

测试契约：
- 主窗只暴露状态控件；明细表通过 ``detail_table()`` 访问。
- ``_switch_status`` 文案前缀 ``当前：未启用`` / ``当前：已启用`` 不变。
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
    QComboBox,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core import quality_probe as qp
from app.core.proof_state import TOPIC_PROBE_OBSERVED, QualityStatsState
from app.models import OcrProject


# ─────────────────────────────────────────────────────────────
# 自绘圆环：抽样字符观察
# ─────────────────────────────────────────────────────────────

class _RatioRing(QWidget):
    """抽样观察圆环：外环按内部 ratio 上色，中心显示等级/状态 + 副标签。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ratio: Optional[float] = None  # None = 未启用 / 样本不足
        self._main_text: str = ""
        self._sub_text: str = ""
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_ratio(self, ratio: Optional[float], sub_text: str = "", main_text: str = "") -> None:
        self._ratio = ratio
        self._sub_text = sub_text
        self._main_text = main_text
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
            center_main = self._main_text or "—"
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
            center_main = self._main_text or "样本观察"

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
        self._summary.setText(f"共 {len(probes)} 个抽样字符/假象字位置（非按页、非全量错误率）")
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
    """抽样字符校对观察主窗（proof UI clarity：紧凑型）。"""

    def __init__(
        self,
        project_provider: Callable[[], Optional[OcrProject]],
        refresh_panels_cb: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("抽样字符校对观察")
        self.setModal(True)
        # 紧凑主窗：360 × 320 足够开关 + 圆环 + 详情入口
        self.resize(360, 320)
        self._project_provider = project_provider
        self._refresh_panels_cb = refresh_panels_cb
        self._detail_dialog: Optional[QualityStatsDetailDialog] = None
        self._quality_state = QualityStatsState(enabled=False)
        self._density_feedback = ""
        self._build_ui()
        self._refresh_view()
        # Round 17 实时刷新：订阅 probe.observed → 每次更正立刻刷新本窗 + gallery
        try:
            from app.core.proof_state_bus import ProofStateBus
            self._unsub_probe = ProofStateBus.instance().subscribe(
                TOPIC_PROBE_OBSERVED, self._on_probe_observed,
            )
        except Exception:
            self._unsub_probe = None

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

        sand_row = QHBoxLayout()
        sand_row.setSpacing(8)
        sand_row.addWidget(QLabel("掺沙密度："))
        self._sand_count_spin = QSpinBox()
        self._sand_count_spin.setRange(0, 1000)
        self._sand_count_spin.setSuffix(" 个")
        self._sand_count_spin.setFixedWidth(90)
        sand_row.addWidget(self._sand_count_spin)
        self._sand_unit_combo = QComboBox()
        self._sand_unit_combo.addItem("每千字", 1000)
        self._sand_unit_combo.addItem("每万字", 10000)
        self._sand_unit_combo.setFixedWidth(100)
        sand_row.addWidget(self._sand_unit_combo)
        self._density_status = QLabel("")
        self._density_status.setStyleSheet("color:#1a73e8;")
        sand_row.addWidget(self._density_status, 1)
        sand_row.addStretch()
        root.addLayout(sand_row)

        # 中部：圆环
        self._ring = _RatioRing()
        root.addWidget(self._ring, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self._scope_note = QLabel("")
        self._scope_note.setWordWrap(True)
        self._scope_note.setStyleSheet("color:#6b7280; font-size:12px;")
        root.addWidget(self._scope_note)

        # 详情入口 + 关闭
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._btn_detail = QPushButton("详情…")
        self._btn_detail.setMinimumHeight(28)
        self._btn_detail.setToolTip("查看每个假象字位置的修正情况")
        self._btn_detail.clicked.connect(self._open_detail)
        bottom.addWidget(self._btn_detail)
        # Round 18：可靠、明确、可触发的刷新入口（不依赖任何 bus / signal 路径）。
        self._btn_refresh = QPushButton("立刻刷新")
        self._btn_refresh.setMinimumHeight(28)
        self._btn_refresh.setToolTip(
            "以 line.final_text 为锚扫一遍所有 probe，重算更正数并刷新本窗。"
        )
        self._btn_refresh.clicked.connect(self._on_manual_refresh)
        bottom.addWidget(self._btn_refresh)
        bottom.addStretch()
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        bottom.addWidget(btn_box)
        root.addLayout(bottom)

        self._load_sampler_controls()
        self._sand_count_spin.valueChanged.connect(self._on_sampler_controls_changed)
        self._sand_unit_combo.currentIndexChanged.connect(self._on_sampler_controls_changed)

    def detail_table(self) -> QTableWidget:
        """Return the detail table hosted by the lazy detail dialog."""
        return self._ensure_detail_dialog()._table

    @property
    def quality_state(self) -> QualityStatsState:
        """Typed quality-probe state consumed by tests and status panels."""
        return self._quality_state

    def _ensure_detail_dialog(self) -> QualityStatsDetailDialog:
        if self._detail_dialog is None:
            self._detail_dialog = QualityStatsDetailDialog(self)
        return self._detail_dialog

    def _load_sampler_controls(self) -> None:
        from app.core.app_config import AppConfig

        cfg = AppConfig.instance()
        try:
            sand_count = int(cfg.get("quality_probe_sand_count", 25))
            sand_unit = int(cfg.get("quality_probe_sand_unit_chars", 1000))
        except (TypeError, ValueError):
            sand_count, sand_unit = 25, 1000
        self._sand_count_spin.blockSignals(True)
        self._sand_unit_combo.blockSignals(True)
        self._sand_count_spin.setValue(max(0, sand_count))
        idx = self._sand_unit_combo.findData(sand_unit)
        self._sand_unit_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._sand_unit_combo.blockSignals(False)
        self._sand_count_spin.blockSignals(False)

    def _save_sampler_controls(self) -> None:
        from app.core.app_config import AppConfig

        cfg = AppConfig.instance()
        cfg.set("quality_probe_sand_count", self._sand_count_spin.value())
        cfg.set("quality_probe_sand_unit_chars", self._sand_unit_combo.currentData())
        self._density_feedback = f"已保存：{self._density_text()}"

    def _density_text(self) -> str:
        return f"{self._sand_count_spin.value()} 个/{self._sand_unit_combo.currentText()}"

    def _refresh_panels(self) -> None:
        try:
            self._refresh_panels_cb()
        except Exception:
            pass

    def _resample_active_store(self) -> bool:
        project = self._project_provider()
        if project is None or not project.pages:
            return False
        cfg = qp.sampler_config_from_app_config()
        store = qp.ProbeSampler(cfg).sample(project)
        if len(store) == 0:
            qp.reset_active_store()
            self._btn_toggle.setChecked(False)
            return False
        qp.set_active_store(store)
        self._density_feedback = (
            f"已保存并生效：{self._density_text()}，投放 {len(store)}/{store.target_probes} 个"
        )
        return True

    def _on_sampler_controls_changed(self, *_args) -> None:
        self._save_sampler_controls()
        if self._btn_toggle.isChecked():
            self._resample_active_store()
            self._refresh_panels()
        self._refresh_view()

    # ── 行为 ───────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        if checked:
            self._save_sampler_controls()
            project = self._project_provider()
            if project is None or not project.pages:
                QMessageBox.information(
                    self, "正确率统计",
                    "尚无校对数据，请先完成 OCR。",
                )
                self._btn_toggle.setChecked(False)
                return
            if not self._resample_active_store():
                QMessageBox.information(
                    self, "正确率统计",
                    "当前正文中没有可投放假象字的位置（页面太短、全是数字/公式/标题，或缺少已有切图字符）。",
                )
                qp.reset_active_store()
                self._btn_toggle.setChecked(False)
                return
        else:
            qp.reset_active_store()
        self._refresh_panels()
        self._refresh_view()

    def _open_detail(self) -> None:
        dlg = self._ensure_detail_dialog()
        dlg.populate()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_probe_observed(self, _payload=None, **_kwargs) -> None:
        """Round 17：probe.observed 事件回调 — 实时刷新评测窗 + 通知 panel 刷新 gallery。"""
        try:
            self._refresh_panels()
        except Exception:
            pass
        try:
            self._refresh_view()
        except Exception:
            pass

    def _on_manual_refresh(self) -> None:
        """Round 18：用户主动点"立刻刷新"——可靠刷新触发点。

        1. 调 ``qp.detect_corrections`` 以 line.final_text 为锚扫一遍所有 pending probe，
           漏报的位置在这里补标 corrected 并广播。
        2. 重新拉一遍 panels 与本窗的 view（圆环 / 详情表）。
        """
        try:
            project = self._project_provider()
            qp.detect_corrections(qp.get_active_store(), project)
        except Exception:
            pass
        try:
            self._refresh_panels()
        except Exception:
            pass
        self._refresh_view()
        if self._detail_dialog is not None:
            try:
                self._detail_dialog.populate()
            except Exception:
                pass

    def closeEvent(self, event):  # type: ignore[override]
        unsub = getattr(self, "_unsub_probe", None)
        if callable(unsub):
            try:
                unsub()
            except Exception:
                pass
            self._unsub_probe = None
        super().closeEvent(event)

    def _refresh_view(self) -> None:
        store = qp.get_active_store()
        if store is None or len(store) == 0:
            self._quality_state = QualityStatsState(
                enabled=False,
                density_text=self._density_feedback or f"当前密度：{self._density_text()}",
            )
            self._btn_toggle.setChecked(False)
            self._btn_toggle.setText("开始统计")
            self._switch_status.setText("当前：未启用")
            self._switch_status.setStyleSheet("color:#666;")
            self._density_status.setText(self._density_feedback or f"当前密度：{self._density_text()}")
            self._scope_note.setText("状态：未启用；修改密度会自动保存。")
            self._ring.set_ratio(None, "")
            self._ensure_detail_dialog().populate()
            return

        n = len(store)
        density_label = "每万字" if getattr(store, "sand_unit_chars", 1000) == 10000 else "每千字"
        sand_count = getattr(store, "sand_count", None)
        density_text = (
            f"掺沙 {sand_count} 个/{density_label}"
            if sand_count is not None else "掺沙密度沿用旧比例"
        )
        self._btn_toggle.setChecked(True)
        self._btn_toggle.setText("停止统计")
        self._switch_status.setText(f"当前：已启用（{n} 个抽样字符）")
        self._switch_status.setStyleSheet("color:#1a73e8;")
        self._density_status.setText(self._density_feedback or f"已生效：{density_text}")
        self._scope_note.setText(
            f"候选池 {getattr(store, 'sampled_from_chars', 0)} 字；目标 {store.target_probes} 个；"
            f"实际投放 {n} 个。"
        )

        rep = qp.score(store)
        self._quality_state = QualityStatsState(
            enabled=True,
            total_probes=rep.total_probes,
            corrected=rep.corrected,
            pending=rep.pending,
            detect_ratio=rep.detect_ratio,
            grade=rep.grade,
            grade_label=rep.grade_label,
            sampled_from_chars=int(getattr(store, "sampled_from_chars", 0)),
            target_probes=int(getattr(store, "target_probes", 0)),
            density_text=density_text,
        )
        # 新设计：probe 只有两种观测状态 — pending / corrected
        total = rep.total_probes
        corrected = rep.corrected
        observed = corrected
        ratio = rep.detect_ratio
        main_text = "观察中" if rep.grade == "INSUFFICIENT" else rep.grade_label
        self._ring.set_ratio(
            ratio,
            f"{density_text} · 已观察 {observed}/{n}",
            main_text,
        )
        # 始终把最新明细同步到详情子窗（隐藏也保持可查看的 detail_table 数据）
        self._ensure_detail_dialog().populate()
