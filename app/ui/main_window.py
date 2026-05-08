"""主窗口：步骤导航栏 + QStackedWidget + 全局工作流控制。

核心变更：
- 业务逻辑委托给 WorkflowController
- UI 只发出用户意图，不直接管理项目状态
- 步骤按钮根据 controller 状态启用
- OCR 完成事件与"进入校对"导航意图严格分离
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QVBoxLayout, QWidget, QFrame,
)

from app.controllers.workflow_controller import (
    WorkflowController, STEP_IMPORT, STEP_LAYOUT, STEP_OCR,
    STEP_HPROOF, STEP_VPROOF,
)
from app.core.logging import get_logger
from app.models import OcrProject, Page
from app.services import ImportService
from app.ui.recognize.import_panel import ImportPanel
from app.ui.recognize.layout_panel import LayoutPanel
from app.ui.proof.h_proof import HProofPanel
from app.ui.proof.v_proof import VProofPanel
from app.ui.export.export_dialog import ExportDialog

logger = get_logger(__name__)


# ── 步骤导航按钮 ───────────────────────────────────────────────
class StepButton(QPushButton):
    def __init__(self, label: str, step: int, parent=None):
        super().__init__(label, parent)
        self.step = step
        self.setObjectName("stepBtn")
        self.setCheckable(True)
        self.setMinimumHeight(32)


# (label, target_step, active_on_steps)  — 导入页不在导航栏内
_NAV_ITEMS = [
    ("①  版面分析",  STEP_LAYOUT, frozenset({STEP_LAYOUT, STEP_OCR})),
    ("②  横向校对",  STEP_HPROOF, frozenset({STEP_HPROOF})),
    ("③  纵向校对",  STEP_VPROOF, frozenset({STEP_VPROOF})),
]


class TopNavBar(QWidget):
    """顶部水平导航：← → + 步骤按钮 + 项目名 + 导出按钮。"""
    step_clicked   = Signal(int)
    prev_clicked   = Signal()
    next_clicked   = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(46)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(4)

        # 品牌 logo
        brand = QLabel("OCR 后处理")
        brand.setStyleSheet(
            "color:#1a73e8; font-size:14px; font-weight:bold; margin-right:8px;"
        )
        layout.addWidget(brand)

        # ← → 箭头
        self._btn_prev = QPushButton("←")
        self._btn_next = QPushButton("→")
        for b in (self._btn_prev, self._btn_next):
            b.setObjectName("ghostBtn")
            b.setFixedSize(30, 30)
        self._btn_prev.setToolTip("上一步")
        self._btn_next.setToolTip("下一步")
        self._btn_prev.clicked.connect(self.prev_clicked)
        self._btn_next.clicked.connect(self.next_clicked)
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_next)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color:#e3e8ef; margin:8px 6px;")
        layout.addWidget(sep)

        # 步骤按钮
        self._buttons: List[StepButton] = []
        for label, target_step, _active in _NAV_ITEMS:
            btn = StepButton(label, target_step)
            btn.setMinimumWidth(100)
            btn.clicked.connect(lambda _, s=target_step: self.step_clicked.emit(s))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # 项目名
        self._project_lbl = QLabel("（无项目）")
        self._project_lbl.setStyleSheet("color:#555; font-size:13px; margin-right:8px;")
        layout.addWidget(self._project_lbl)

        # 导出按钮
        self._btn_export = QPushButton("⤓ 导出")
        self._btn_export.setObjectName("ghostBtn")
        self._btn_export.clicked.connect(self.export_clicked)
        layout.addWidget(self._btn_export)

    def set_active(self, step: int) -> None:
        for i, (_, _, active_set) in enumerate(_NAV_ITEMS):
            self._buttons[i].setChecked(step in active_set)

    def set_enabled_up_to(self, max_step: int) -> None:
        for i, (_, target_step, _) in enumerate(_NAV_ITEMS):
            self._buttons[i].setEnabled(target_step <= max_step)

    def set_project_name(self, name: str) -> None:
        self._project_lbl.setText(name)


# ── 主窗口 ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._controller = WorkflowController()
        self._current_step: int = STEP_IMPORT

        self.setWindowTitle("OCR 后处理")
        _screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(_screen.width() * 0.85), int(_screen.height() * 0.85))
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._top_nav.set_enabled_up_to(self._controller.max_step)
        self._go_to_step(STEP_IMPORT)

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 顶部导航栏
        self._top_nav = TopNavBar()
        self._top_nav.step_clicked.connect(self._on_step_clicked)
        self._top_nav.prev_clicked.connect(self._prev_step)
        self._top_nav.next_clicked.connect(self._next_step)
        self._top_nav.export_clicked.connect(self._show_export_dialog)
        outer.addWidget(self._top_nav)

        # 中：QStackedWidget
        self._stack = QStackedWidget()

        self._import_panel  = ImportPanel()
        self._layout_panel  = LayoutPanel()
        self._ocr_placeholder = QLabel("正在 OCR 识别，请稍候…")
        self._ocr_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ocr_placeholder.setStyleSheet("font-size:18px; color:#888;")
        self._hproof_panel  = HProofPanel()
        self._vproof_panel  = VProofPanel()

        for w in (
            self._import_panel, self._layout_panel, self._ocr_placeholder,
            self._hproof_panel, self._vproof_panel,
        ):
            self._stack.addWidget(w)

        outer.addWidget(self._stack, 1)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

    def _connect_signals(self) -> None:
        """连接所有信号，包括 controller 和面板之间的信号。"""

        # ----- 面板信号 -> Controller -----

        # 导入面板：图片/PDF 准备就绪
        self._import_panel.images_ready.connect(self._on_images_ready)

        # 版面面板：运行分析按钮 -> 启动分析
        self._layout_panel.run_button.clicked.connect(self._start_layout_analysis)

        # 校对面板：保存修改
        self._hproof_panel.proof_saved.connect(self._auto_save)
        self._vproof_panel.proof_saved.connect(self._auto_save)

        # ----- Controller 信号 -> UI -----

        self._controller.project_changed.connect(self._on_project_changed)
        self._controller.step_enabled_changed.connect(self._top_nav.set_enabled_up_to)
        self._controller.step_requested.connect(self._go_to_step)
        self._controller.ocr_finished.connect(self._on_ocr_finished)
        self._controller.layout_finished.connect(self._on_layout_finished)
        self._controller.worker_error.connect(self._on_worker_error)
        self._controller.status_message.connect(self._status_bar.showMessage)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_m = menu.addMenu("文件(&F)")
        act_new  = QAction("新建项目(&N)", self)
        act_open = QAction("打开项目(&O)…", self)
        act_save = QAction("保存项目(&S)", self)
        act_quit = QAction("退出(&Q)", self)
        act_new.triggered.connect(self._new_project)
        act_open.triggered.connect(self._open_project)
        act_save.triggered.connect(self._save_project)
        act_quit.triggered.connect(self.close)
        for a in (act_new, act_open, act_save, None, act_quit):
            if a is None:
                file_m.addSeparator()
            else:
                file_m.addAction(a)

        export_m = menu.addMenu("导出(&E)")
        act_export = QAction("导出…", self)
        act_export.triggered.connect(self._show_export_dialog)
        export_m.addAction(act_export)

        settings_m = menu.addMenu("设置(&T)")
        act_ocr_cfg = QAction("OCR 引擎设置…", self)
        act_ocr_cfg.triggered.connect(self._show_ocr_settings)
        settings_m.addAction(act_ocr_cfg)

        help_m = menu.addMenu("帮助(&H)")
        act_about = QAction("关于", self)
        act_about.triggered.connect(lambda: QMessageBox.about(
            self, "关于", "OCR 后处理软件 v0.2.0\n基于 PaddleOCR + PySide6"
        ))
        help_m.addAction(act_about)

    # ── 步骤切换 ───────────────────────────────────────────────

    def _go_to_step(self, step: int) -> None:
        """直接跳转到步骤（不经过 controller 校验）。"""
        self._stack.setCurrentIndex(step)
        self._top_nav.set_active(step)
        self._current_step = step

    def _on_step_clicked(self, step: int) -> None:
        """用户点击步骤栏按钮 → 让 controller 判断是否允许跳转。"""
        self._controller.request_step(step)

    def _prev_step(self) -> None:
        if self._current_step > STEP_LAYOUT:
            self._controller.request_step(self._current_step - 1)

    def _next_step(self) -> None:
        self._controller.request_step(self._current_step + 1)

    # ── Controller 回调 ─────────────────────────────────────────

    def _on_project_changed(self, project: OcrProject) -> None:
        self._top_nav.set_project_name(f"项目：{project.name}")

    def _on_layout_finished(self, pages: List[Page]) -> None:
        """版面分析完成，更新 UI 并自动启动 OCR。"""
        self._layout_panel.show_analysis_result(pages)
        self._layout_panel.run_button.setEnabled(True)
        self._status_bar.showMessage(f"版面分析完成：{len(pages)} 页")
        self._start_ocr()

    def _on_ocr_finished(self, pages: List[Page]) -> None:
        """OCR 完成（由 controller 发出，业务事件）。"""
        self._hproof_panel.load_pages(pages)
        self._vproof_panel.load_pages(pages)
        self._go_to_step(STEP_HPROOF)

    def _on_worker_error(self, msg: str) -> None:
        """Worker 出错时恢复所有按钮状态并显示错误。"""
        self._layout_panel.run_button.setEnabled(True)
        if hasattr(self._layout_panel, '_btn_ocr'):
            self._layout_panel._btn_ocr.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理失败：\n{msg}")

    # ── 文件操作 ───────────────────────────────────────────────

    def _new_project(self) -> None:
        name, ok = self._prompt_project_name()
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目文件", f"{name}.ocrproj", "OCR 项目 (*.ocrproj)"
        )
        if not path:
            return
        if self._controller.new_project(name, path):
            self._go_to_step(STEP_IMPORT)
            self._import_panel.reset()

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目文件", "", "OCR 项目 (*.ocrproj)"
        )
        if not path:
            return
        if self._controller.open_project(path):
            project = self._controller.project
            if project and project.pages:
                self._layout_panel.set_pages(project.pages)
                if all(p.is_analyzed for p in project.pages):
                    self._layout_panel.show_analysis_result(project.pages)
                if project.ocr_completed:
                    self._hproof_panel.load_pages(project.pages)
                    self._vproof_panel.load_pages(project.pages)
            self._go_to_step(self._controller.get_open_step())

    def _save_project(self) -> None:
        if not self._controller.project:
            QMessageBox.information(self, "提示", "当前无项目，请先新建或打开项目")
            return
        self._controller.save_project()

    def _auto_save(self) -> None:
        self._controller.auto_save()

    # ── 工作流事件处理 ──────────────────────────────────────────

    def _on_images_ready(self, paths: List[str]) -> None:
        """导入文件 → 使用 ImportService 创建 Page 对象 → 交给 controller。"""
        project = self._controller.project
        if not project:
            self._controller.ensure_transient_project("未命名项目")

        try:
            from pathlib import Path
            cache_dir = Path(self._controller._project.db_path or ".").parent / ".cache"
            importer = ImportService(cache_dir=cache_dir)
            result = importer.import_paths(paths)

            if not result.pages:
                QMessageBox.warning(
                    self, "导入失败",
                    "所有文件导入失败，详见日志。\n" +
                    "\n".join(f"• {p}: {r}" for p, r in result.failed[:5])
                )
                return

            self._controller.on_images_ready(result.pages)

            if result.failed:
                fail_msg = "\n".join(f"• {Path(p).name}: {r}" for p, r in result.failed[:3])
                self._status_bar.showMessage(
                    f"导入完成：{result.success_count} 页成功，{result.failed_count} 个失败"
                )
                QMessageBox.information(
                    self, "导入完成",
                    f"成功导入 {result.success_count} 页。\n"
                    f"{result.failed_count} 个文件失败：\n{fail_msg}"
                )
            else:
                self._status_bar.showMessage(f"导入 {result.success_count} 页")

        except Exception as e:
            logger.error("Import failed: %s", e)
            QMessageBox.critical(self, "导入错误", f"导入过程出错：{e}")
            return

        self._layout_panel.set_pages(result.pages)
        self._go_to_step(STEP_LAYOUT)

    def _start_layout_analysis(self) -> None:
        if not self._controller.project or not self._controller.project.pages:
            return
        self._layout_panel.run_button.setEnabled(False)
        if not self._controller.start_layout_analysis(self._controller.project.pages):
            self._layout_panel.run_button.setEnabled(True)

    def _start_ocr(self) -> None:
        """OCR 启动（版面分析完成后自动触发，后台运行，不跳转页面）。"""
        if not self._controller.project or not self._controller.project.pages:
            return
        pages = self._controller.project.pages
        if self._controller.get_recognizable_block_count() == 0:
            return  # 无可识别块，静默跳过
        self._status_bar.showMessage("正在 OCR 识别…")
        self._controller.start_ocr(pages)

    # ── 导出 ────────────────────────────────────────────────────

    def _show_export_dialog(self) -> None:
        project = self._controller.project
        if not project or not project.pages:
            QMessageBox.information(self, "提示", "请先完成 OCR 识别再导出")
            return

        # 导出前检查
        summary = project.get_export_summary()
        warnings = []
        if summary["unrecognized_blocks"] > 0:
            warnings.append(f"{summary['unrecognized_blocks']} 个块未识别")
        if summary["unproofed_lines"] > 0:
            warnings.append(f"{summary['unproofed_lines']} 行未校对")
        if summary["flagged_lines"] > 0:
            warnings.append(f"{summary['flagged_lines']} 行低置信度")

        if warnings:
            msg = "导出前请注意：\n" + "\n".join(f"  • {w}" for w in warnings)
            msg += "\n\n是否继续导出？"
            if QMessageBox.question(self, "导出确认", msg) != QMessageBox.StandardButton.Yes:
                return

        dlg = ExportDialog(project, self)
        dlg.exec()

    # ── 辅助 ───────────────────────────────────────────────────

    def _prompt_project_name(self) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称：", text="新项目")
        return name.strip() or "新项目", ok

    def closeEvent(self, event) -> None:
        self._controller.close()
        super().closeEvent(event)

    def _show_ocr_settings(self) -> None:
        from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        from app.core.ocr_config import get_config
        from app.engines.real_ocr_adapter import get_engine_description
        dlg = ApiSettingsDialog(self)
        # 防止对话框超出屏幕高度
        _avail = QApplication.primaryScreen().availableGeometry()
        dlg.setMaximumHeight(int(_avail.height() * 0.88))
        if dlg.exec():
            cfg = get_config()
            self._status_bar.showMessage(f"OCR 引擎：{get_engine_description(cfg['mode'])}")
