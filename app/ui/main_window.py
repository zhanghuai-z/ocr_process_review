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

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QStackedWidget,
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
from app.ui.widgets.nav_rail import NavRail
from app.ui.widgets.top_bar import TopBar

logger = get_logger(__name__)


# ── 步骤导航按钮 ───────────────────────────────────────────────
class StepButton(QPushButton):
    def __init__(self, label: str, step: int, parent=None):
        super().__init__(label, parent)
        self.step = step
        self.setObjectName("stepBtn")
        self.setCheckable(True)
        self.setMinimumHeight(32)


# ── OCR 进度占位面板 ────────────────────────────────────────────
class _OcrProgressWidget(QWidget):
    """OCR 进行中占位面板：显示进度条和页面计数。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self._title = QLabel("正在 OCR 识别，请稍候…")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-size:18px; color:#555;")

        self._bar = QProgressBar()
        self._bar.setFixedWidth(320)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 0)  # 默认不确定模式

        self._count = QLabel("")
        self._count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count.setStyleSheet("font-size:13px; color:#888;")

        layout.addStretch()
        layout.addWidget(self._title)
        layout.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._count)
        layout.addStretch()

    def reset(self) -> None:
        self._bar.setRange(0, 0)
        self._count.setText("")

    @Slot(int, int)
    def update_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(current + 1)
            self._count.setText(f"{current + 1} / {total} 页")


# (label, target_step, active_on_steps)  — 导入页不在导航栏内
_NAV_ITEMS = [
    ("①  版面分析",  STEP_LAYOUT, frozenset({STEP_LAYOUT, STEP_OCR})),
    ("②  横向校对",  STEP_HPROOF, frozenset({STEP_HPROOF})),
    ("③  纵向校对",  STEP_VPROOF, frozenset({STEP_VPROOF})),
]

# TopBar 面包屑用的步骤名（不含 ①②③ 前缀）
_STEP_BREADCRUMB = {
    STEP_IMPORT: "导入",
    STEP_LAYOUT: "版面分析",
    STEP_OCR:    "OCR 识别",
    STEP_HPROOF: "横向校对",
    STEP_VPROOF: "纵向校对",
}


class TopNavBar(QWidget):
    """顶部水平导航：步骤按钮 + 版面分析启动 + 项目名 + 导出按钮。"""
    step_clicked   = Signal(int)
    layout_run_clicked = Signal()
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

        # 步骤按钮
        self._buttons: List[StepButton] = []
        for label, target_step, _active in _NAV_ITEMS:
            btn = StepButton(label, target_step)
            btn.setMinimumWidth(100)
            btn.clicked.connect(lambda _, s=target_step: self.step_clicked.emit(s))
            self._buttons.append(btn)
            layout.addWidget(btn)

        self._btn_run_layout = QPushButton("▶")
        self._btn_run_layout.setToolTip("运行版面分析")
        self._btn_run_layout.setFixedSize(34, 30)
        self._btn_run_layout.setEnabled(False)
        self._btn_run_layout.setStyleSheet(
            "QPushButton { background:#22c55e; color:white; border:0; border-radius:4px; "
            "font-size:16px; font-weight:bold; }"
            "QPushButton:disabled { background:#b8dec5; color:#f3fff6; }"
        )
        self._btn_run_layout.clicked.connect(self.layout_run_clicked)
        layout.addWidget(self._btn_run_layout)

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

    def set_layout_run_enabled(self, enabled: bool) -> None:
        self._btn_run_layout.setEnabled(enabled)

    def set_project_name(self, name: str) -> None:
        self._project_lbl.setText(name)


# ── 主窗口 ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._controller = WorkflowController()
        # 注意：_current_step / _current_page_number 的 ownership 已收到
        # WorkflowController；此处不再持有镜像。MainWindow 通过 controller 的
        # current_step_changed / current_page_number_changed signal 同步 UI。

        self.setWindowTitle("OCR 后处理")
        _screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(_screen.width() * 0.85), int(_screen.height() * 0.85))
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._nav_rail.set_enabled_up_to(self._controller.max_step)
        self._go_to_step(STEP_IMPORT)

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # 根布局：左 NavRail | 右（TopBar / Stack）
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 左：竖向导航栏
        self._nav_rail = NavRail()
        self._nav_rail.step_clicked.connect(self._on_step_clicked)
        self._nav_rail.settings_clicked.connect(self._show_ocr_settings)
        outer.addWidget(self._nav_rail)

        # 右：内容列
        right_col = QWidget()
        right_v = QVBoxLayout(right_col)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(0)

        # 顶部 TopBar（项目名 + 运行 + 导出）
        self._top_bar = TopBar()
        self._top_bar.layout_run_clicked.connect(self._start_layout_analysis)
        self._top_bar.export_clicked.connect(self._show_export_dialog)
        right_v.addWidget(self._top_bar)

        # 中：QStackedWidget
        self._stack = QStackedWidget()

        self._import_panel  = ImportPanel()
        self._layout_panel  = LayoutPanel()
        self._ocr_placeholder = _OcrProgressWidget()
        self._hproof_panel  = HProofPanel()
        self._vproof_panel  = VProofPanel()

        for w in (
            self._import_panel, self._layout_panel, self._ocr_placeholder,
            self._hproof_panel, self._vproof_panel,
        ):
            self._stack.addWidget(w)

        right_v.addWidget(self._stack, 1)

        outer.addWidget(right_col, 1)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

    def _connect_signals(self) -> None:
        """连接所有信号，包括 controller 和面板之间的信号。"""

        # ----- 面板信号 -> Controller -----

        # 导入面板：图片/PDF 准备就绪
        self._import_panel.images_ready.connect(self._on_images_ready)

        # 校对面板：保存修改
        self._hproof_panel.proof_saved.connect(self._auto_save)
        self._vproof_panel.proof_saved.connect(self._auto_save)

        # ----- Controller 信号 -> UI -----

        # 让 controller 拥有 proof 同步 ownership（merge vs load 决策 + line_count 跟踪）
        self._controller.register_proof_panels(self._hproof_panel, self._vproof_panel)
        self._controller.project_changed.connect(self._on_project_changed)
        self._controller.step_enabled_changed.connect(self._nav_rail.set_enabled_up_to)
        # view-state signals: controller 是 ownership 持有者，MainWindow 只订阅
        self._controller.current_step_changed.connect(self._on_current_step_changed)
        self._controller.current_page_number_changed.connect(self._on_current_page_number_changed)
        self._controller.layout_run_enabled_changed.connect(self._top_bar.set_layout_run_enabled)
        self._controller.step_requested.connect(self._go_to_step)
        self._controller.ocr_finished.connect(self._on_ocr_finished)
        self._controller.layout_finished.connect(self._on_layout_finished)
        self._controller.layout_progress.connect(self._layout_panel.update_analysis_progress)
        self._controller.ocr_progress.connect(self._on_ocr_progress)
        self._controller.worker_error.connect(self._on_worker_error)
        self._controller.status_message.connect(self._status_bar.showMessage)
        self._layout_panel.geometry_changed.connect(self._controller.save_project)
        self._layout_panel.page_selected.connect(self._on_layout_page_selected)
        self._hproof_panel.page_selected.connect(self._on_hproof_page_selected)

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
        # Phase 11：原 h_proof 工具栏「评测开关 + 报告」入口迁移到这里。
        act_quality_stats = QAction("正确率统计…", self)
        act_quality_stats.triggered.connect(self._show_quality_stats)
        settings_m.addAction(act_quality_stats)

        help_m = menu.addMenu("帮助(&H)")
        act_about = QAction("关于", self)
        act_about.triggered.connect(lambda: QMessageBox.about(
            self, "关于", "OCR 后处理软件 v0.2.0\n基于 PaddleOCR + PySide6"
        ))
        help_m.addAction(act_about)

        # 视图：主题切换
        view_m = menu.addMenu("视图(&V)")
        from app.core.app_config import AppConfig
        from app.ui.styles import apply_theme, available_themes
        current_theme = str(AppConfig.instance().get("theme") or "light").lower()
        if current_theme == "dark_teal":
            current_theme = "dark"
        from PySide6.QtGui import QActionGroup
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for name in available_themes():
            label = {"light": "浅色主题", "dark": "深色主题"}.get(name, name)
            act = QAction(label, self, checkable=True)
            act.setChecked(name == current_theme)
            act.triggered.connect(lambda _checked, n=name: self._switch_theme(n))
            theme_group.addAction(act)
            view_m.addAction(act)

    def _switch_theme(self, name: str) -> None:
        from app.core.app_config import AppConfig
        from app.ui.styles import apply_theme
        app = QApplication.instance()
        if app is None:
            return
        actual = apply_theme(app, name)
        AppConfig.instance().set("theme", actual)
        self._status_bar.showMessage(f"已切换到 {actual} 主题", 3000)

    # ── 步骤切换 ───────────────────────────────────────────────

    def _go_to_step(self, step: int) -> None:
        """直接跳转到步骤（不经过 controller 校验）。

        controller 是 current_step 的唯一真值。这里通过 set_current_step 通知
        controller，由 current_step_changed signal 驱动 UI 同步（_on_current_step_changed）。
        proof / layout 的副作用（采样 / 刷新 / 设置 page_number）也只读
        controller.current_page_number，避免两边状态错位。
        """
        self._controller.set_current_step(step)
        if step in (STEP_HPROOF, STEP_VPROOF):
            self._controller.ensure_quality_probe_sampled()
            if step == STEP_HPROOF:
                self._hproof_panel.set_current_page_number(
                    self._controller.current_page_number)
            self._controller.refresh_proof_quality_probe_state(step)
        elif step == STEP_LAYOUT:
            self._layout_panel.set_current_page_number(
                self._controller.current_page_number)

    def _on_current_step_changed(self, step: int) -> None:
        """controller.current_step_changed → 同步 stack 和顶部栏激活态。"""
        self._stack.setCurrentIndex(step)
        self._nav_rail.set_active(step)
        self._top_bar.set_step_name(_STEP_BREADCRUMB.get(step, ""))

    def _on_current_page_number_changed(self, page_number: int) -> None:
        """controller.current_page_number_changed → 同步两个相关面板。"""
        # h_proof / layout 都需要知道当前页（v_proof 用自己的 gallery 选择）
        self._hproof_panel.set_current_page_number(page_number)
        self._layout_panel.set_current_page_number(page_number)

    def _on_step_clicked(self, step: int) -> None:
        """用户点击步骤栏按钮 → 让 controller 判断是否允许跳转。"""
        self._controller.request_step(step)

    def _prev_step(self) -> None:
        cur = self._controller.current_step
        if cur > STEP_LAYOUT:
            self._controller.request_step(cur - 1)

    def _next_step(self) -> None:
        self._controller.request_step(self._controller.current_step + 1)

    def _on_layout_page_selected(self, idx: int) -> None:
        page_number = self._controller.page_number_at(idx)
        if page_number is not None:
            # ownership 在 controller，set_current_page_number 会 emit signal
            # 由 _on_current_page_number_changed 同步两个面板。
            self._controller.set_current_page_number(page_number)

    def _on_hproof_page_selected(self, page_number: int) -> None:
        # controller 持有 ownership；signal 会回到 _on_current_page_number_changed
        # 同步 layout/hproof（hproof 自己已经发起，重复 setText no-op）。
        self._controller.set_current_page_number(page_number)

    # ── Controller 回调 ─────────────────────────────────────────

    def _on_project_changed(self, project: OcrProject) -> None:
        self._top_bar.set_project_name(project.name)
        # 新项目载入：复位状态徽章为「未运行版面分析」
        self._top_bar.set_status("idle", "未运行")

    def _on_layout_finished(self, pages: List[Page]) -> None:
        """版面分析完成，更新 UI；后续 OCR 由 WorkflowController 调度。"""
        self._layout_panel.show_analysis_result(pages)
        self._layout_panel.run_button.setEnabled(True)
        self._controller.set_layout_run_enabled(True)
        failed = sum(1 for page in pages if page.error_message)
        if failed:
            self._top_bar.set_status("warn", f"完成 {len(pages) - failed}/{len(pages)}")
            self._status_bar.showMessage(
                f"版面分析完成：{len(pages) - failed}/{len(pages)} 页成功，{failed} 页失败"
            )
        else:
            self._top_bar.set_status("done", "已运行")
            self._status_bar.showMessage(f"版面分析完成：{len(pages)} 页")

    def _on_ocr_finished(self, pages: List[Page]) -> None:
        """OCR 完成（由 controller 发出，业务事件）。

        proof 面板同步（merge vs load + line_count 维护）的 ownership 已收到
        controller 内部。OCR 完成只开放校对入口，不再强制把用户带到横校。"""
        self._controller.sync_proof_panels()

    def _on_ocr_progress(self, progress) -> None:
        if progress.total_pages > 0:
            current = max(0, min(progress.completed_pages - 1, progress.total_pages - 1))
            self._ocr_placeholder.update_progress(current, progress.total_pages)
        # proof 同步 ownership 在 controller；这里只发触发
        self._controller.sync_proof_panels()

    def _on_worker_error(self, msg: str) -> None:
        """Worker 出错时恢复所有按钮状态并显示错误。"""
        self._layout_panel.run_button.setEnabled(True)
        self._controller.set_layout_run_enabled(True)
        if hasattr(self._layout_panel, '_btn_ocr'):
            self._layout_panel._btn_ocr.setEnabled(True)
        if self._controller.current_step in (STEP_LAYOUT, STEP_OCR) or "版面分析" in msg:
            self._layout_panel.finish_analysis_progress(msg)
            self._top_bar.set_status("warn", "失败")
            self._status_bar.showMessage(f"处理失败：{msg}")
            return
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
            self._controller.reset_proof_sync_state()
            self._go_to_step(STEP_IMPORT)
            self._import_panel.reset()

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目文件", "", "OCR 项目 (*.ocrproj)"
        )
        if not path:
            return
        if self._controller.open_project(path):
            self._controller.reset_proof_sync_state()
            if self._controller.has_pages:
                pages = self._controller.pages
                self._layout_panel.set_pages(pages)
                self._controller.set_layout_run_enabled(True)
                if self._controller.is_fully_analyzed:
                    self._layout_panel.show_analysis_result(pages)
                if self._controller.ocr_completed:
                    # 全量初始化 proof 面板（force_load=True 跳过 merge 路径）
                    self._controller.sync_proof_panels(force_load=True)
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
            # 用 controller.cache_dir 取代 self._controller._project.db_path 私有访问
            cache_dir = self._controller.cache_dir
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
            self._controller.reset_proof_sync_state()
            self._controller.set_layout_run_enabled(True)

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
        if not self._controller.has_pages:
            return
        self._layout_panel.run_button.setEnabled(False)
        self._controller.set_layout_run_enabled(False)
        self._layout_panel.start_analysis_progress(len(self._controller.pages))
        self._top_bar.set_status("running", "运行中…")
        if not self._controller.start_layout_analysis(self._controller.pages):
            self._layout_panel.finish_analysis_progress("版面分析未启动")
            self._layout_panel.run_button.setEnabled(True)
            self._controller.set_layout_run_enabled(True)
            self._top_bar.set_status("idle", "未运行")

    def _start_ocr(self) -> None:
        """OCR 启动（版面分析完成后自动触发）：跳转到 OCR 进度页并显示进度。"""
        if not self._controller.has_pages:
            return
        pages = self._controller.pages
        if self._controller.get_recognizable_block_count() == 0:
            return  # 无可识别块，静默跳过
        self._ocr_placeholder.reset()
        self._go_to_step(STEP_OCR)
        self._status_bar.showMessage("正在 OCR 识别…")
        self._controller.start_ocr(pages, notify_page_callback=self._ocr_placeholder.update_progress)

    # ── 导出 ────────────────────────────────────────────────────

    def _show_export_dialog(self) -> None:
        if not self._controller.has_pages:
            QMessageBox.information(self, "提示", "请先完成 OCR 识别再导出")
            return
        project = self._controller.project

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

    def _show_quality_stats(self) -> None:
        """打开『正确率统计』对话框。

        - project 通过 lambda 延迟取，避免 dialog 持过期引用
        - 启用/关闭后回调到 panel 的 refresh_quality_probe_state，让显示和 active store 一致
        """
        from app.ui.widgets.quality_stats_dialog import QualityStatsDialog
        dlg = QualityStatsDialog(
            project_provider=lambda: self._controller.project,
            refresh_panels_cb=self._refresh_proof_panels_after_quality_toggle,
            parent=self,
        )
        dlg.exec()

    def _refresh_proof_panels_after_quality_toggle(self) -> None:
        """正确率统计开关切换后，把 active store 的变化同步到两个 proof 面板。"""
        for panel in (self._hproof_panel, self._vproof_panel):
            fn = getattr(panel, "refresh_quality_probe_state", None)
            if callable(fn):
                fn()

    def _show_ocr_settings(self) -> None:
        from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        from app.core.ocr_config import get_config
        from app.engines.real_ocr_adapter import get_engine_description
        dlg = ApiSettingsDialog(self)
        if dlg.exec():
            cfg = get_config()
            self._status_bar.showMessage(f"OCR 引擎：{get_engine_description(cfg['mode'])}")
