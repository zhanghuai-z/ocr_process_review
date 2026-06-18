"""主窗口：顶部流程条 + QStackedWidget + 全局工作流控制。

核心变更：
- 业务逻辑委托给 WorkflowController
- UI 只发出用户意图，不直接管理项目状态
- 顶部流程按钮根据 controller 状态启用
- OCR 完成事件与"进入校对"导航意图严格分离
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from app.controllers.workflow_controller import (
    WorkflowController, STEP_IMPORT, STEP_LAYOUT, STEP_OCR,
    STEP_HPROOF, STEP_VPROOF,
)
from app.core.workflow_state import WorkflowViewState
from app.core.logging import get_logger
from app.models import OcrProject, Page
from app.services import ImportService
from app.ui.recognize.import_panel import ImportPanel
from app.ui.recognize.layout_panel import LayoutPanel
from app.ui.proof.h_proof import HProofPanel
from app.ui.proof.v_proof import VProofPanel
from app.ui.export.export_dialog import ExportDialog
from app.ui.widgets.top_bar import TopBar

logger = get_logger(__name__)


# ── OCR 状态栏进度 ──────────────────────────────────────────────
class _OcrProgressWidget(QWidget):
    """状态栏边缘进度：不覆盖版面工作区。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(6)

        self._title = QLabel("OCR")
        self._title.setStyleSheet("font-size:12px; color:#555;")

        self._bar = QProgressBar()
        self._bar.setFixedWidth(140)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 0)  # 默认不确定模式

        self._count = QLabel("")
        self._count.setStyleSheet("font-size:12px; color:#888;")

        layout.addWidget(self._title)
        layout.addWidget(self._bar)
        layout.addWidget(self._count)
        self.hide()

    def reset(self) -> None:
        self._bar.setRange(0, 0)
        self._count.setText("准备中")
        self.show()

    def finish(self) -> None:
        self.hide()

    @Slot(int, int)
    def update_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(current + 1)
            self._count.setText(f"{current + 1}/{total}")
            self.show()


# TopBar 面包屑用的步骤名（不含 ①②③ 前缀）
_STEP_BREADCRUMB = {
    STEP_IMPORT: "导入",
    STEP_LAYOUT: "版面分析",
    STEP_OCR:    "OCR 识别",
    STEP_HPROOF: "横向校对",
    STEP_VPROOF: "纵向校对",
}


class _CurrentPageStack(QStackedWidget):
    """QStackedWidget variant whose size hint follows the visible page only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _idx: self.updateGeometry())

    def sizeHint(self):
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        return current.minimumSizeHint() if current is not None else super().minimumSizeHint()


class ImportWorker(QThread):
    """Render/import files off the UI thread."""

    all_done = Signal(object)
    error = Signal(str)

    def __init__(self, paths: List[str], cache_dir: Path, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._cache_dir = cache_dir

    def run(self) -> None:
        try:
            importer = ImportService(cache_dir=self._cache_dir)
            self.all_done.emit(importer.import_paths(self._paths))
        except Exception as exc:
            logger.error("Import worker failed: %s", exc)
            self.error.emit(str(exc))


# ── 主窗口 ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._controller = WorkflowController()
        # 注意：_current_step / _current_page_number 的 ownership 已收到
        # WorkflowController；此处不再持有镜像。MainWindow 通过 controller 的
        # WorkflowViewState typed signal 同步 UI。
        self._last_workflow_view_state: Optional[WorkflowViewState] = None
        self._workbench_initial_resize_done = False
        self._last_proof_sync_completed_pages = 0
        self._import_worker: ImportWorker | None = None

        self.setWindowTitle("OCR 后处理")
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._top_bar.set_enabled_up_to(self._controller.max_step)
        self._go_to_step(STEP_IMPORT)
        self._resize_for_initial_import_page()

    def _resize_for_initial_import_page(self) -> None:
        target_w, target_h = 960, 540
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_w = min(target_w, max(320, available.width() - 80))
            target_h = min(target_h, max(320, available.height() - 80))
        self.resize(target_w, target_h)

    def _resize_for_initial_workbench_page(self) -> None:
        if self._workbench_initial_resize_done or self.isMaximized() or self.isFullScreen():
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            self._set_centered_window_size(max(self.width(), 1280), max(self.height(), 720))
            self._workbench_initial_resize_done = True
            return
        available = screen.availableGeometry()
        target_w = min(
            max(self.width(), int(available.width() * 0.80)),
            max(320, available.width() - 40),
        )
        target_h = min(
            max(self.height(), int(available.height() * 0.80)),
            max(320, available.height() - 40),
        )
        self._set_centered_window_size(target_w, target_h)
        self._workbench_initial_resize_done = True

    def _set_centered_window_size(self, target_w: int, target_h: int) -> None:
        min_hint = self.minimumSizeHint()
        target_w = max(target_w, min_hint.width())
        target_h = max(target_h, min_hint.height())
        geometry = self.geometry()
        frame = self.frameGeometry()
        center = frame.center()
        left_frame = max(0, geometry.left() - frame.left())
        top_frame = max(0, geometry.top() - frame.top())
        right_frame = max(0, frame.right() - geometry.right())
        bottom_frame = max(0, frame.bottom() - geometry.bottom())
        frame_w = target_w + left_frame + right_frame
        frame_h = target_h + top_frame + bottom_frame
        frame_x = center.x() - frame_w // 2
        frame_y = center.y() - frame_h // 2

        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            max_frame_x = available.right() - frame_w + 1
            max_frame_y = available.bottom() - frame_h + 1
            frame_x = (
                available.left()
                if max_frame_x < available.left()
                else max(available.left(), min(frame_x, max_frame_x))
            )
            frame_y = (
                available.top()
                if max_frame_y < available.top()
                else max(available.top(), min(frame_y, max_frame_y))
            )

        self.setGeometry(
            frame_x + left_frame,
            frame_y + top_frame,
            target_w,
            target_h,
        )

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # 根布局：TopBar / Stack。步骤入口统一收在顶部。
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶部 TopBar（项目名 + 流程入口 + 运行 + 导出）
        self._top_bar = TopBar()
        self._top_bar.step_clicked.connect(self._on_step_clicked)
        self._top_bar.layout_run_clicked.connect(self._start_layout_analysis)
        self._top_bar.export_clicked.connect(self._show_export_dialog)
        root.addWidget(self._top_bar)

        # 中：QStackedWidget
        self._stack = _CurrentPageStack()

        self._import_panel  = ImportPanel()
        self._layout_panel  = LayoutPanel()
        self._ocr_placeholder = _OcrProgressWidget()
        self._hproof_panel  = HProofPanel()
        self._vproof_panel  = VProofPanel()

        for w in (
            self._import_panel, self._layout_panel,
            self._hproof_panel, self._vproof_panel,
        ):
            self._stack.addWidget(w)
        self._stack_widget_by_step = {
            STEP_IMPORT: self._import_panel,
            STEP_LAYOUT: self._layout_panel,
            STEP_OCR: self._layout_panel,
            STEP_HPROOF: self._hproof_panel,
            STEP_VPROOF: self._vproof_panel,
        }

        root.addWidget(self._stack, 1)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.addWidget(self._ocr_placeholder)
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
        # typed view-state signal: controller 是 ownership 持有者，MainWindow 只消费 WorkflowViewState
        self._controller.view_state_changed.connect(self._on_workflow_view_state_changed)
        self._controller.step_requested.connect(self._go_to_step)
        self._controller.ocr_finished.connect(self._on_ocr_finished)
        self._controller.layout_finished.connect(self._on_layout_finished)
        self._controller.layout_progress.connect(self._layout_panel.update_analysis_progress)
        self._controller.ocr_progress.connect(self._on_ocr_progress)
        self._controller.worker_error.connect(self._on_worker_error)
        self._controller.status_message.connect(self._status_bar.showMessage)
        self._layout_panel.geometry_changed.connect(self._controller.save_project)
        self._layout_panel.block_contract_changed.connect(self._controller.handle_block_contract_changed)
        self._layout_panel.ocr_entry_requested.connect(self._controller.handle_ocr_entry_requested)
        self._layout_panel.page_selected.connect(self._on_layout_page_selected)
        self._controller.focus_page.connect(self._layout_panel.set_current_page_number)
        self._controller.page_gate_state.connect(self._layout_panel.set_page_gate_state)
        self._controller.primary_action.connect(self._layout_panel.set_primary_action)
        self._hproof_panel.page_selected.connect(self._on_hproof_page_selected)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_m = menu.addMenu("文件(&F)")
        act_open = QAction("打开项目(&O)…", self)
        act_save = QAction("保存项目(&S)", self)
        act_close_project = QAction("关闭项目(&W)", self)
        act_close_project.setShortcut(QKeySequence("Ctrl+W"))
        act_open.triggered.connect(self._open_project)
        act_save.triggered.connect(self._save_project)
        act_close_project.triggered.connect(self._close_project)
        for a in (act_open, act_save, None, act_close_project):
            if a is None:
                file_m.addSeparator()
            else:
                file_m.addAction(a)

        export_m = menu.addMenu("导出(&E)")
        act_export = QAction("导出…", self)
        act_export.triggered.connect(self._show_export_dialog)
        export_m.addAction(act_export)

        settings_m = menu.addMenu("设置(&T)")
        act_find = QAction("查找版面块…", self)
        act_find.setShortcut(QKeySequence.StandardKey.Find)
        act_find.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_find.triggered.connect(self._show_layout_find)
        settings_m.addAction(act_find)
        settings_m.addSeparator()
        act_ocr_cfg = QAction("OCR 引擎设置…", self)
        act_ocr_cfg.triggered.connect(self._show_ocr_settings)
        settings_m.addAction(act_ocr_cfg)
        act_clear_charocr_cache = QAction("清空 CharOCR 缓存", self)
        act_clear_charocr_cache.triggered.connect(self._clear_charocr_cache)
        settings_m.addAction(act_clear_charocr_cache)
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
        widget = self._stack_widget_by_step.get(step, self._layout_panel)
        self._stack.setCurrentWidget(widget)
        self._top_bar.set_active(step)
        self._top_bar.set_step_name(_STEP_BREADCRUMB.get(step, ""))
        if step != STEP_IMPORT:
            self._resize_for_initial_workbench_page()

    def _on_current_page_number_changed(self, page_number: int) -> None:
        """controller.current_page_number_changed → 只同步当前可见重面板。"""
        step = self._controller.current_step
        if step == STEP_HPROOF:
            self._hproof_panel.set_current_page_number(page_number)
        elif step in (STEP_LAYOUT, STEP_OCR):
            self._layout_panel.set_current_page_number(page_number)

    def _on_workflow_view_state_changed(self, state: WorkflowViewState) -> None:
        """controller WorkflowViewState → 同步 workflow 相关 UI。"""
        prev = self._last_workflow_view_state
        if prev is None or prev.max_step != state.max_step:
            self._top_bar.set_enabled_up_to(state.max_step)
        if prev is None or prev.current_step != state.current_step:
            self._on_current_step_changed(state.current_step)
        if prev is None or prev.current_page_number != state.current_page_number:
            self._on_current_page_number_changed(state.current_page_number)
        if prev is None or prev.layout_run_enabled != state.layout_run_enabled:
            self._top_bar.set_layout_run_enabled(state.layout_run_enabled)
        self._last_workflow_view_state = state

    def _on_step_clicked(self, step: int) -> None:
        """用户点击步骤栏按钮 → 让 controller 判断是否允许跳转。"""
        self._controller.request_step(step)

    def _prev_step(self) -> None:
        cur = self._controller.current_step
        if cur > STEP_LAYOUT:
            self._controller.request_step(cur - 1)

    def _next_step(self) -> None:
        self._controller.request_step(self._controller.current_step + 1)

    def _on_layout_page_selected(self, page_number: int) -> None:
        # ownership 在 controller，set_current_page_number 会 emit signal
        # 由 _on_current_page_number_changed 同步两个面板。
        self._controller.set_current_page_number(page_number)

    def _on_hproof_page_selected(self, page_number: int) -> None:
        # controller 持有 ownership；signal 会回到 _on_current_page_number_changed
        # 同步 layout/hproof（hproof 自己已经发起，重复 setText no-op）。
        self._controller.set_current_page_number(page_number)

    # ── Controller 回调 ─────────────────────────────────────────

    def _on_project_changed(self, project: Optional[OcrProject]) -> None:
        if project is None:
            self._top_bar.set_project_name("")
            self._top_bar.set_status("hidden")
            return
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
        self._ocr_placeholder.finish()
        self._last_proof_sync_completed_pages = len(pages)
        self._controller.sync_proof_panels()
        self._layout_panel.refresh_text_indexes()

    def _on_ocr_progress(self, progress) -> None:
        if progress.total_blocks <= 0 and progress.completed_pages <= 0:
            self._ocr_placeholder.reset()
        elif progress.total_blocks > 0:
            if progress.current_block <= 0:
                self._ocr_placeholder.reset()
            else:
                current = max(0, min(progress.current_block - 1, progress.total_blocks - 1))
                self._ocr_placeholder.update_progress(current, progress.total_blocks)
        elif progress.total_pages > 0:
            current = max(0, min(progress.completed_pages - 1, progress.total_pages - 1))
            self._ocr_placeholder.update_progress(current, progress.total_pages)
        completed_pages = max(0, int(getattr(progress, "completed_pages", 0) or 0))
        if completed_pages == 0 and int(getattr(progress, "current_page", 0) or 0) <= 1:
            self._last_proof_sync_completed_pages = 0
        if completed_pages <= self._last_proof_sync_completed_pages:
            return
        self._last_proof_sync_completed_pages = completed_pages
        if self._controller.current_step in (STEP_HPROOF, STEP_VPROOF):
            self._controller.sync_proof_panels()

    def _on_worker_error(self, msg: str) -> None:
        """Worker 出错时恢复所有按钮状态并显示错误。"""
        self._layout_panel.run_button.setEnabled(True)
        self._controller.set_layout_run_enabled(True)
        if hasattr(self._layout_panel, '_btn_ocr'):
            self._layout_panel._btn_ocr.setEnabled(True)
        if self._controller.current_step in (STEP_LAYOUT, STEP_OCR) or "版面分析" in msg:
            self._ocr_placeholder.finish()
            self._layout_panel.finish_analysis_progress(msg)
            self._top_bar.set_status("warn", "失败")
            self._status_bar.showMessage(f"处理失败：{msg}")
            return
        QMessageBox.critical(self, "错误", f"处理失败：\n{msg}")

    # ── 文件操作 ───────────────────────────────────────────────

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
                if self._controller.has_any_ocr_result:
                    # 全量初始化 proof 面板（force_load=True 跳过 merge 路径）
                    self._controller.sync_proof_panels(force_load=True)
            self._go_to_step(self._controller.get_open_step())

    def _save_project(self) -> None:
        if not self._controller.project:
            QMessageBox.information(self, "提示", "当前无项目，请先导入或打开项目")
            return
        if not self._controller.store:
            self._save_project_as()
            return
        self._controller.save_project()

    def _save_project_as(self) -> bool:
        project = self._controller.project
        if not project:
            return False
        default_name = f"{project.name or '未命名项目'}.ocrproj"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目文件", default_name, "OCR 项目 (*.ocrproj)"
        )
        if not path:
            return False
        return self._controller.save_project_as(path)

    def _confirm_close_project_save(self) -> bool:
        if not self._controller.project:
            return True
        result = QMessageBox.question(
            self,
            "关闭项目",
            "关闭当前项目前是否保存？\n选择“取消”会保留当前项目。",
            (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Save,
        )
        if result == QMessageBox.StandardButton.Cancel:
            return False
        if result == QMessageBox.StandardButton.Save and not self._controller.save_project():
            QMessageBox.warning(self, "保存失败", "项目未保存，已取消关闭。")
            return False
        return True

    def _close_project(self) -> None:
        if not self._controller.project:
            self._status_bar.showMessage("当前没有打开的项目")
            return
        if self._controller.has_running_workers() or self._import_worker_is_running():
            QMessageBox.warning(self, "关闭项目", "后台任务仍在运行，请等待完成后再关闭项目。")
            return
        if not self._confirm_close_project_save():
            return
        if self._controller.close_project():
            self._reset_workspace_after_project_closed()

    def _reset_workspace_after_project_closed(self) -> None:
        self._workbench_initial_resize_done = False
        self._ocr_placeholder.finish()
        self._import_panel.reset()
        self._layout_panel.reset()
        self._hproof_panel.reset()
        self._vproof_panel.reset()
        self._top_bar.set_project_name("")
        self._top_bar.set_status("hidden")
        self._stack.setCurrentWidget(self._import_panel)
        self._status_bar.showMessage("项目已关闭")

    def _auto_save(self) -> None:
        self._controller.auto_save()

    # ── 工作流事件处理 ──────────────────────────────────────────

    def _on_images_ready(self, paths: List[str]) -> None:
        """导入文件 → 使用 ImportService 创建 Page 对象 → 交给 controller。"""
        if self._import_worker_is_running():
            self._status_bar.showMessage("文件仍在导入中…")
            return
        project = self._controller.project
        if not project:
            self._controller.ensure_transient_project("未命名项目")

        try:
            cache_dir = self._controller.cache_dir
            self._import_panel.setEnabled(False)
            self._status_bar.showMessage(f"正在导入 {len(paths)} 个文件…")
            self._import_worker = ImportWorker(paths, cache_dir, self)
            self._import_worker.all_done.connect(self._on_import_done)
            self._import_worker.error.connect(self._on_import_error)
            self._import_worker.finished.connect(self._on_import_finished)
            self._import_worker.start()
        except Exception as e:
            logger.error("Import failed: %s", e)
            self._import_panel.setEnabled(True)
            QMessageBox.critical(self, "导入错误", f"导入过程出错：{e}")

    def _import_worker_is_running(self) -> bool:
        worker = self._import_worker
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    def _on_import_done(self, result) -> None:
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
        self._layout_panel.set_pages(result.pages)
        self._go_to_step(STEP_LAYOUT)

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

    def _on_import_error(self, message: str) -> None:
        QMessageBox.critical(self, "导入错误", f"导入过程出错：{message}")

    def _on_import_finished(self) -> None:
        self._import_panel.setEnabled(True)
        self._import_worker = None

    def _start_layout_analysis(self) -> None:
        if not self._controller.has_pages:
            return
        self._layout_panel.run_button.setEnabled(False)
        self._controller.set_layout_run_enabled(False)
        self._layout_panel.start_analysis_progress(len(self._controller.pages))
        self._top_bar.set_status("running", "运行中…")
        QApplication.processEvents()
        if not self._controller.start_layout_analysis(self._controller.pages):
            self._layout_panel.finish_analysis_progress("版面分析未启动")
            self._layout_panel.run_button.setEnabled(True)
            self._controller.set_layout_run_enabled(True)
            self._top_bar.set_status("idle", "未运行")

    def _start_ocr(self) -> None:
        """OCR 启动（版面分析完成后自动触发）：保留版面工作区，仅在状态栏显示进度。"""
        if not self._controller.has_pages:
            return
        if self._controller.is_hanwang_mode():
            self._controller.handle_ocr_entry_requested("main_window", self._controller.current_page_number)
            return
        pages = self._controller.pages
        if self._controller.get_recognizable_block_count() == 0:
            return  # 无可识别块，静默跳过
        self._ocr_placeholder.reset()
        self._last_proof_sync_completed_pages = 0
        self._go_to_step(STEP_OCR)
        self._status_bar.showMessage("正在文字识别…")
        QApplication.processEvents()
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

    def closeEvent(self, event) -> None:
        if self._controller.has_running_workers() or self._import_worker_is_running():
            QMessageBox.warning(self, "关闭程序", "后台任务仍在运行，请等待完成后再关闭。")
            event.ignore()
            return
        if self._controller.project:
            if self._controller.store:
                self._controller.save_project()
            else:
                result = QMessageBox.question(
                    self,
                    "保存项目",
                    "当前项目尚未保存。退出前是否保存？",
                    (
                        QMessageBox.StandardButton.Save
                        | QMessageBox.StandardButton.Discard
                        | QMessageBox.StandardButton.Cancel
                    ),
                    QMessageBox.StandardButton.Save,
                )
                if result == QMessageBox.StandardButton.Cancel:
                    event.ignore()
                    return
                if result == QMessageBox.StandardButton.Save and not self._save_project_as():
                    event.ignore()
                    return
        self._controller.close()
        super().closeEvent(event)

    def _clear_charocr_cache(self) -> None:
        path = self._controller.active_charocr_cache_dir
        if QMessageBox.question(
            self,
            "清空 CharOCR 缓存",
            f"将删除当前项目的 CharOCR 原生识别缓存：\n{path}\n\n继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        cleared = self._controller.clear_charocr_cache()
        self._status_bar.showMessage(f"CharOCR 缓存已清空：{cleared}", 5000)

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

    def _show_layout_find(self) -> None:
        if not self._controller.has_pages:
            self._status_bar.showMessage("当前没有可查找的版面块", 3000)
            return
        self._go_to_step(STEP_LAYOUT)
        self._layout_panel.show_find_dialog()

    def _show_ocr_settings(self) -> None:
        from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        from app.core.app_config import get_config
        from app.engines.real_ocr_adapter import get_engine_description
        dlg = ApiSettingsDialog(self)
        if dlg.exec():
            cfg = get_config()
            self._status_bar.showMessage(f"OCR 引擎：{get_engine_description(cfg['mode'])}")
