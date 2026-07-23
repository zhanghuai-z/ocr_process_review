"""Qt shell for the session-scoped OCR workbench."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.controllers.workflow_controller import (
    STEP_HPROOF,
    STEP_IMPORT,
    STEP_LAYOUT,
    STEP_OCR,
    STEP_VPROOF,
    WorkflowController,
)
from app.application import ImportCompletionView
from app.core.workflow_state import WorkflowProgressState, WorkflowViewState
from app.ui.export.export_dialog import ExportDialog
from app.ui.recognize.import_panel import ImportPanel
from app.ui.recognize.layout_panel import LayoutPanel
from app.ui.recognize.ocr_panel import OcrPanel
from app.ui.proof.h_proof import HProofPanel
from app.ui.proof.v_proof import VProofPanel
from app.ui.widgets.top_bar import TopBar


class _ShellProgress(QWidget):
    """Compact status-bar progress projection for layout and OCR runs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._active = False
        self._title = QLabel("进度")
        self._title.setFixedWidth(58)
        # 状态栏左侧已有运行描述（如"正在运行版面分析"），进度条只留 bar+计数
        self._title.setVisible(False)
        self._detail = QLabel("")
        self._detail.setMinimumWidth(96)
        # 阶段描述不进界面：进度条只表达"在进行/走到哪"，细节收进悬浮提示
        self._detail.setVisible(False)
        self._count = QLabel("")
        self._count.setFixedWidth(62)
        self._count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedWidth(150)
        self._bar.setFixedHeight(16)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._detail)
        layout.addWidget(self._bar)
        layout.addWidget(self._count)
        self.hide()

    @property
    def active(self) -> bool:
        return self._active

    def _show_progress(self, title: str, detail: str, count: str, value: int) -> None:
        self._active = True
        self._title.setText(title)
        self._detail.setText(detail)
        self._count.setText(count)
        self._bar.setValue(max(0, min(100, int(value))))
        tip = " · ".join(part for part in (title, detail, count) if part)
        self._bar.setToolTip(tip)
        self._title.setToolTip(tip)
        self._count.setToolTip(tip)
        self.show()

    def start_layout(self, total: int) -> None:
        total = max(1, int(total))
        self._show_progress("版面分析", "提交请求", f"0/{total} 页", 0)

    def update_layout(self, current: int, total: int) -> None:
        total = max(1, int(total))
        done = max(0, min(int(current), total))
        self._show_progress(
            "版面分析",
            "解析结果" if done >= total else "等待结果",
            f"{done}/{total} 页",
            round(done / total * 100),
        )

    def update_layout_stage(self, current: int, total: int, message: str) -> None:
        total = max(1, int(total))
        page = max(1, min(int(current), total))
        completed = max(0, min(page - 1, total - 1))
        self._show_progress(
            "版面分析",
            message or "处理中",
            f"{page}/{total} 页",
            round((completed + 0.05) / total * 100),
        )

    def update_ocr(self, progress: WorkflowProgressState) -> None:
        total_pages = max(1, progress.total_pages)
        completed = max(0, min(progress.completed_pages, total_pages))
        message = str(progress.message or "")
        if message == "已完成":
            page_fraction = 0.0
            page = max(1, completed)
        else:
            page_fraction = self._ocr_page_fraction(progress)
            page = min(total_pages, completed + 1)
        value = round((completed + page_fraction) / total_pages * 100)
        self._show_progress(
            "OCR", self._ocr_stage_label(message), f"{page}/{total_pages} 页", value
        )

    @staticmethod
    def _ocr_stage_label(message: str) -> str:
        lowered = message.lower()
        if "行框" in message or "pp-ocr" in lowered or "prepass" in lowered:
            return "行框定位"
        if "路由" in message:
            return "路由编译"
        if "segimg" in lowered or "分块" in message:
            return "字符切分"
        if "recog" in lowered and "准备" in message:
            return "识别准备"
        if "识别" in message or "hanwang" in lowered:
            return "字符识别"
        if "完成" in message:
            return "写回结果"
        return "准备中"

    @staticmethod
    def _ocr_page_fraction(progress: WorkflowProgressState) -> float:
        message = str(progress.message or "")
        lowered = message.lower()
        if "行框定位" in message:
            return 0.10
        if "行框完成" in message:
            return 0.18
        if "路由" in message:
            return 0.26
        if "segimg" in lowered or "分块" in message:
            return 0.36
        if "recog" in lowered and "准备" in message:
            return 0.44
        if progress.total > 0 and ("识别" in message or "hanwang" in lowered):
            ratio = max(0.0, min(1.0, progress.current / progress.total))
            return 0.44 + ratio * 0.52
        return 0.05

    def finish(self) -> None:
        self._active = False
        self.hide()


class MainWindow(QMainWindow):
    """Application shell that communicates with the controller by records and UIDs."""

    def __init__(self, controller: WorkflowController | None = None) -> None:
        super().__init__()
        self._controller = controller or WorkflowController(self)
        self._last_view_state: WorkflowViewState | None = None
        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self.resize(1280, 800)
        self._controller.publish_state()

    # ------------------------------------------------------------------ UI setup

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._top_bar = TopBar()
        self._top_bar.step_clicked.connect(self._on_step_clicked)
        self._top_bar.layout_run_clicked.connect(self._start_layout_analysis)
        self._top_bar.export_clicked.connect(self._show_export_dialog)
        root.addWidget(self._top_bar)

        self._stack = QStackedWidget()
        self._import_panel = ImportPanel()
        self._layout_panel = LayoutPanel()
        self._ocr_panel = OcrPanel()
        self._hproof_panel = HProofPanel()
        self._vproof_panel = VProofPanel()
        self._stack.addWidget(self._import_panel)
        self._stack.addWidget(self._layout_panel)
        self._stack.addWidget(self._ocr_panel)
        self._stack.addWidget(self._hproof_panel)
        self._stack.addWidget(self._vproof_panel)
        self._stack_by_step = {
            STEP_IMPORT: self._import_panel,
            STEP_LAYOUT: self._layout_panel,
            STEP_OCR: self._layout_panel,
            STEP_HPROOF: self._hproof_panel,
            STEP_VPROOF: self._vproof_panel,
        }
        root.addWidget(self._stack, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        self._progress = _ShellProgress()
        status.addPermanentWidget(self._progress)
        status.showMessage("就绪")

    def _connect_signals(self) -> None:
        self._import_panel.images_ready.connect(self._start_import)
        self._import_panel.open_project_requested.connect(self._open_project)
        self._layout_panel.page_selected.connect(self._select_page)
        self._layout_panel.layout_edit_requested.connect(self._apply_layout_edit)
        self._layout_panel.ocr_entry_requested.connect(self._on_ocr_entry_requested)
        self._layout_panel.analysis_cancel_requested.connect(
            self._controller.cancel_layout_analysis
        )
        self._ocr_panel.page_selected.connect(self._select_page)
        self._ocr_panel.ocr_requested.connect(self._start_ocr)
        self._ocr_panel.go_to_proof_requested.connect(
            lambda: self._controller.request_step(STEP_HPROOF)
        )
        self._hproof_panel.proof_edit_requested.connect(self._apply_proof_edit)
        self._vproof_panel.proof_edit_requested.connect(self._apply_proof_edit)
        self._controller.session_identity_changed.connect(self._on_identity_changed)
        self._controller.layout_workspace_changed.connect(self._on_layout_workspace_changed)
        self._controller.ocr_workspace_changed.connect(self._on_ocr_workspace_changed)
        self._controller.proof_workspace_changed.connect(self._on_proof_workspace_changed)
        self._controller.proof_workspace_patched.connect(self._on_proof_workspace_patched)
        self._controller.import_finished.connect(self._on_import_finished)
        self._controller.layout_finished.connect(self._on_layout_finished)
        self._controller.layout_progress.connect(self._on_layout_progress)
        self._controller.layout_stage.connect(self._on_layout_stage)
        self._controller.layout_cancelled.connect(self._on_layout_cancelled)
        self._controller.ocr_finished.connect(self._on_ocr_finished)
        self._controller.ocr_progress.connect(self._on_ocr_progress)
        self._controller.ocr_cancelled.connect(self._on_ocr_cancelled)
        self._controller.worker_error.connect(self._on_worker_error)
        self._controller.status_message.connect(self._set_status_message)
        self._controller.step_requested.connect(self._go_to_step)
        self._controller.view_state_changed.connect(self._on_view_state_changed)
        self._controller.page_gate_state.connect(self._layout_panel.set_page_gate_state)
        self._controller.primary_action.connect(self._layout_panel.set_primary_action)

    def _build_menu(self) -> None:
        self.menuBar().hide()

        file_menu = QMenu("文件", self)
        save = QAction("保存项目(&S)", self)
        save.setShortcut(QKeySequence.StandardKey.Save)
        save_as = QAction("另存为…", self)
        save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        close = QAction("关闭项目(&W)", self)
        close.setShortcut(QKeySequence("Ctrl+W"))
        save.triggered.connect(self._save_project)
        save_as.triggered.connect(self._save_project_as_dialog)
        close.triggered.connect(self._close_project)
        file_menu.addAction(save)
        file_menu.addAction(save_as)
        file_menu.addSeparator()
        file_menu.addAction(close)
        self.addAction(save)
        self.addAction(save_as)
        self.addAction(close)

        more_menu = QMenu("更多", self)
        settings = QAction("设置…", self)
        settings.triggered.connect(self._show_ocr_settings)
        more_menu.addAction(settings)
        more_menu.addSeparator()
        describe = QAction("OCR 服务状态", self)
        describe.triggered.connect(
            lambda: self._set_status_message(self._controller.ocr_engine_description(), 5000)
        )
        more_menu.addAction(describe)
        self._top_bar.set_menus(file_menu, more_menu)

    # ------------------------------------------------------------------ workflow navigation

    def _on_view_state_changed(self, state: WorkflowViewState) -> None:
        previous = self._last_view_state
        self._top_bar.set_enabled_up_to(state.max_step)
        self._top_bar.set_layout_run_enabled(state.layout_run_enabled)
        if previous is None or previous.current_step != state.current_step:
            self._go_to_step(state.current_step)
        if previous is None or previous.current_page_uid != state.current_page_uid:
            if state.current_page_uid:
                self._layout_panel.set_current_page_uid(state.current_page_uid)
                self._ocr_panel.set_current_page_uid(state.current_page_uid)
        self._last_view_state = state

    def _on_step_clicked(self, step: int) -> None:
        self._controller.request_step(step)

    def _go_to_step(self, step: int) -> None:
        if step not in self._stack_by_step:
            raise ValueError(f"unknown workflow step: {step}")
        self._stack.setCurrentWidget(self._stack_by_step[step])
        self._top_bar.setVisible(step != STEP_IMPORT)
        self._top_bar.set_active(step)
        if step != STEP_IMPORT and self._controller.current_step != step:
            self._controller.set_current_step(step)

    def _select_page(self, page_uid: str) -> None:
        self._controller.set_current_page_uid(page_uid)

    # ------------------------------------------------------------------ controller events

    def _on_identity_changed(self, _project_uid: str, name: str) -> None:
        self._top_bar.set_project_name(name)

    def _on_layout_workspace_changed(self, workspace: object) -> None:
        if workspace is None:
            self._layout_panel.reset()
            return
        self._layout_panel.set_workspace(workspace)

    def _on_ocr_workspace_changed(self, workspace: object) -> None:
        if workspace is None:
            self._ocr_panel.reset()
            self._layout_panel.set_ocr_workspace(None)
            return
        self._ocr_panel.set_workspace(workspace)
        self._layout_panel.set_ocr_workspace(workspace)

    def _on_proof_workspace_changed(self, workspace: object) -> None:
        if workspace is None:
            self._hproof_panel.clear_workspace()
            self._vproof_panel.clear_workspace()
            return
        self._hproof_panel.set_workspace(workspace)
        self._vproof_panel.set_workspace(workspace)

    def _on_proof_workspace_patched(self, patch: object) -> None:
        self._hproof_panel.apply_workspace_patch(patch)
        self._vproof_panel.apply_workspace_patch(patch)

    def _on_import_finished(self, result: ImportCompletionView) -> None:
        self._import_panel.setEnabled(True)
        if not result.page_uids:
            failures = "\n".join(
                f"• {failure.source_path}: {failure.error}"
                for failure in result.failures[:5]
            )
            QMessageBox.warning(self, "导入失败", failures or "没有导入页面")
            return
        self._set_status_message(f"导入 {result.success_count} 页")
        if result.failures:
            self._set_status_message(
                f"导入完成：{result.success_count} 页成功，{result.failure_count} 个失败"
            )
        self._controller.request_step(STEP_LAYOUT)

    def _on_layout_progress(self, current: int, total: int) -> None:
        self._progress.update_layout(current, total)
        self._layout_panel.update_analysis_progress(current, total)
        self._set_status_message("版面分析处理中")

    def _on_layout_stage(self, page_uid: str, current: int, total: int, message: str) -> None:
        self._progress.update_layout_stage(current, total, message)
        self._layout_panel.update_analysis_stage(message)
        self._set_status_message(f"{message}：{page_uid}")

    def _on_layout_finished(self) -> None:
        self._progress.finish()
        self._layout_panel.finish_analysis_progress("版面分析完成")
        self._set_status_message("版面分析完成")
        self._controller.request_step(STEP_OCR)

    def _on_layout_cancelled(self) -> None:
        self._progress.finish()
        self._layout_panel.finish_analysis_progress("版面分析已取消")
        self._set_status_message("版面分析已取消")

    def _on_ocr_progress(self, progress: WorkflowProgressState) -> None:
        self._progress.update_ocr(progress)
        self._ocr_panel.on_progress(progress)

    def _on_ocr_finished(self) -> None:
        self._progress.finish()
        self._ocr_panel.finish_progress("OCR 完成")
        self._set_status_message("文字识别完成")

    def _on_ocr_cancelled(self) -> None:
        self._progress.finish()
        self._ocr_panel.finish_progress("OCR 已取消")

    def _load_snapshot_into_panel(self) -> None:
        self._controller.publish_state()

    def _on_worker_error(self, message: str) -> None:
        self._progress.finish()
        self._layout_panel.finish_analysis_progress(message)
        self._import_panel.setEnabled(True)
        self._ocr_panel.finish_progress(message)
        self._set_status_message(f"处理失败：{message}")

    # ------------------------------------------------------------------ application actions

    def _start_import(self, paths: list[str]) -> None:
        if self._controller.has_running_workers():
            self._set_status_message("当前仍有任务运行")
            return
        try:
            self._import_panel.setEnabled(False)
            self._set_status_message(f"正在导入 {len(paths)} 个文件…")
            self._controller.start_import(paths)
        except Exception as exc:
            self._import_panel.setEnabled(True)
            self._on_worker_error(str(exc))

    def _start_layout_analysis(self) -> None:
        if not self._controller.has_pages:
            return
        try:
            self._progress.start_layout(len(self._controller.pages))
            self._layout_panel.start_analysis_progress(len(self._controller.pages))
            self._controller.start_layout_analysis()
            self._set_status_message("正在运行版面分析…")
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _apply_layout_edit(self, command: object) -> None:
        try:
            self._controller.apply_layout_edit(command)
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _apply_proof_edit(self, command: object) -> None:
        try:
            self._controller.apply_proof_edit(command)
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _start_ocr(self) -> None:
        self._start_ocr_for_pages(None)

    def _on_ocr_entry_requested(self, _source: str, page_uid: str) -> None:
        self._start_ocr_for_pages((page_uid,))

    def _start_ocr_for_pages(self, page_uids: tuple[str, ...] | None) -> None:
        try:
            self._controller.start_ocr(page_uids)
            self._set_status_message("正在运行 OCR…")
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _open_project(self) -> None:
        if not self._confirm_save_before_discard("打开项目"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开项目文件",
            "",
            "OCR 项目 (*.ocrproj)",
        )
        if not path:
            return
        try:
            self._controller.open_project(path)
            self._go_to_step(self._controller.get_open_step())
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _save_project(self) -> bool:
        if not self._controller.has_project:
            QMessageBox.information(self, "提示", "当前无项目")
            return False
        try:
            if self._controller.is_bound_project:
                self._controller.save_project()
            else:
                return self._save_project_as_dialog()
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._set_status_message("项目已保存")
        return True

    def _save_project_as_dialog(self) -> bool:
        if not self._controller.has_project:
            return False
        default_name = f"{self._controller.project_name or '未命名项目'}.ocrproj"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存项目文件",
            default_name,
            "OCR 项目 (*.ocrproj)",
        )
        if not path:
            return False
        try:
            self._controller.save_project_as(path)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self._set_status_message("项目已保存")
        return True

    def _show_export_dialog(self) -> None:
        if not self._controller.has_pages:
            QMessageBox.information(self, "提示", "请先导入页面")
            return
        try:
            snapshot = self._controller.capture_export_snapshot()
        except Exception as exc:
            QMessageBox.warning(self, "无法导出", str(exc))
            return
        ExportDialog(snapshot, self).exec()

    def _show_ocr_settings(self) -> None:
        from app.ui.widgets.api_settings_dialog import ApiSettingsDialog

        dialog = ApiSettingsDialog(self)
        if dialog.exec():
            self._set_status_message("OCR 设置已更新", 3000)

    def _confirm_save_before_discard(self, title: str) -> bool:
        if not self._controller.has_project or not self._controller.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            title,
            "当前项目有未保存的修改。",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        return reply == QMessageBox.StandardButton.Discard

    def _close_project(self) -> None:
        if not self._confirm_save_before_discard("关闭项目"):
            return
        try:
            if self._controller.close_project():
                self._reset_workspace()
        except Exception as exc:
            self._on_worker_error(str(exc))

    def _reset_workspace(self) -> None:
        self._import_panel.reset()
        self._layout_panel.reset()
        self._ocr_panel.reset()
        self._hproof_panel.clear_workspace()
        self._vproof_panel.clear_workspace()
        self._top_bar.set_project_name("")
        self._go_to_step(STEP_IMPORT)
        self._set_status_message("项目已关闭")

    def _set_status_message(self, message: str, timeout: int = 0) -> None:
        self.statusBar().showMessage(message, timeout)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._confirm_save_before_discard("关闭程序"):
            event.ignore()
            return
        try:
            self._controller.close()
        except Exception as exc:
            QMessageBox.warning(self, "关闭失败", str(exc))
            event.ignore()
            return
        event.accept()


__all__ = ["MainWindow"]
