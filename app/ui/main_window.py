"""主窗口：步骤导航栏 + QStackedWidget + 全局工作流控制。"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QVBoxLayout, QWidget,
)

from app.models import OcrProject, Page
from app.core.project_store import ProjectStore
from app.core.proof_engine import ProofEngine
from app.ui.recognize.import_panel import ImportPanel
from app.ui.recognize.layout_panel import LayoutPanel
from app.ui.recognize.ocr_panel import OcrPanel
from app.ui.proof.h_proof import HProofPanel
from app.ui.proof.v_proof import VProofPanel
from app.ui.export.export_dialog import ExportDialog


# ── 步骤常量 ──────────────────────────────────────────────────
STEP_IMPORT  = 0
STEP_LAYOUT  = 1
STEP_OCR     = 2
STEP_HPROOF  = 3
STEP_VPROOF  = 4
STEP_EXPORT  = 5  # 导出不是单独页面，通过对话框触发


# ── 版面分析 Worker ────────────────────────────────────────────
class LayoutWorker(QThread):
    page_done = Signal(int, int)
    all_done  = Signal(list)   # List[Page]
    error     = Signal(str)

    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self._pages = pages

    def run(self) -> None:
        try:
            from app.core.layout_analyzer import LayoutAnalyzer
            analyzer = LayoutAnalyzer()
            total = len(self._pages)
            for i, page in enumerate(self._pages):
                analyzer.analyze(page)
                self.page_done.emit(i, total)
            self.all_done.emit(self._pages)
        except Exception as e:
            self.error.emit(str(e))


# ── 步骤导航按钮 ───────────────────────────────────────────────
class StepButton(QPushButton):
    def __init__(self, label: str, step: int, parent=None):
        super().__init__(label, parent)
        self.step = step
        self.setCheckable(True)
        self.setMinimumWidth(90)
        self.setStyleSheet("""
            QPushButton {
                border: none; border-radius: 6px;
                padding: 8px 12px; font-size: 13px;
                color: #aaa; background: transparent;
            }
            QPushButton:checked {
                background: #37373d; color: #fff; font-weight: bold;
            }
            QPushButton:hover:!checked { background: #2d2d30; color: #ccc; }
        """)


class StepBar(QWidget):
    step_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: List[StepButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(2)

        steps = [
            ("① 导入", STEP_IMPORT),
            ("② 版面", STEP_LAYOUT),
            ("③ OCR",  STEP_OCR),
            ("④ 横校", STEP_HPROOF),
            ("⑤ 纵校", STEP_VPROOF),
        ]
        for label, step in steps:
            btn = StepButton(label, step)
            btn.clicked.connect(lambda _, s=step: self.step_clicked.emit(s))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def set_active(self, step: int) -> None:
        for btn in self._buttons:
            btn.setChecked(btn.step == step)

    def set_enabled_up_to(self, max_step: int) -> None:
        for btn in self._buttons:
            btn.setEnabled(btn.step <= max_step)


# ── 主窗口 ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._project: Optional[OcrProject] = None
        self._store: Optional[ProjectStore] = None
        self._layout_worker: Optional[LayoutWorker] = None
        self._ocr_worker = None
        self._proof_engine = ProofEngine()

        self.setWindowTitle("OCR 后处理")
        self.resize(1280, 800)
        self._build_ui()
        self._build_menu()
        self._go_to_step(STEP_IMPORT)

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶部：步骤栏 + 项目名 + 导出按钮
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background:#252526; border-bottom:1px solid #3c3c3c;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 12, 0)

        self._step_bar = StepBar()
        self._step_bar.step_clicked.connect(self._go_to_step)
        h_layout.addWidget(self._step_bar)

        self._project_lbl = QLabel("（无项目）")
        self._project_lbl.setStyleSheet("color:#666; font-size:12px;")
        h_layout.addWidget(self._project_lbl)

        btn_export = QPushButton("导出…")
        btn_export.setStyleSheet("padding:4px 12px;")
        btn_export.clicked.connect(self._show_export_dialog)
        h_layout.addWidget(btn_export)

        root.addWidget(header)

        # 中：QStackedWidget
        self._stack = QStackedWidget()

        self._import_panel  = ImportPanel()
        self._layout_panel  = LayoutPanel()
        self._ocr_panel     = OcrPanel()
        self._hproof_panel  = HProofPanel()
        self._vproof_panel  = VProofPanel()

        for w in (
            self._import_panel, self._layout_panel, self._ocr_panel,
            self._hproof_panel, self._vproof_panel,
        ):
            self._stack.addWidget(w)

        root.addWidget(self._stack)

        # 信号连接
        self._import_panel.images_ready.connect(self._on_images_ready)
        self._layout_panel.analysis_confirmed.connect(self._start_ocr)
        self._layout_panel.run_button.clicked.connect(self._start_layout_analysis)
        self._ocr_panel.recognition_done.connect(self._on_ocr_done)
        self._hproof_panel.proof_saved.connect(self._auto_save)
        self._vproof_panel.proof_saved.connect(self._auto_save)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

    def _build_menu(self) -> None:
        menu = self.menuBar()
        menu.setStyleSheet("background:#252526; color:#ccc;")

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
            self, "关于", "OCR 后处理软件 v0.1.0\n基于 PaddleOCR + PySide6"
        ))
        help_m.addAction(act_about)

    # ── 步骤切换 ───────────────────────────────────────────────

    def _go_to_step(self, step: int) -> None:
        self._stack.setCurrentIndex(step)
        self._step_bar.set_active(step)

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
        if self._store:
            self._store.close()
        self._store = ProjectStore(path)
        self._store.open()
        self._project = OcrProject(name=name, db_path=path)
        self._project = self._store.save_project(self._project)
        self._project_lbl.setText(f"项目：{name}")
        self._step_bar.set_enabled_up_to(STEP_IMPORT)
        self._go_to_step(STEP_IMPORT)
        self._status_bar.showMessage(f"新建项目：{path}")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目文件", "", "OCR 项目 (*.ocrproj)"
        )
        if not path:
            return
        if self._store:
            self._store.close()
        self._store = ProjectStore(path)
        self._store.open()
        self._project = self._store.load_project(project_id=1)
        if not self._project:
            QMessageBox.warning(self, "错误", "项目文件无效或为空")
            return
        self._project_lbl.setText(f"项目：{self._project.name}")
        self._step_bar.set_enabled_up_to(STEP_VPROOF)

        if self._project.pages:
            self._layout_panel.set_pages(self._project.pages)
            if all(p.is_analyzed for p in self._project.pages):
                self._layout_panel.show_analysis_result(self._project.pages)
                self._ocr_panel.set_pages(self._project.pages)
                self._hproof_panel.load_pages(self._project.pages)
                self._vproof_panel.load_pages(self._project.pages)

        self._go_to_step(STEP_LAYOUT)
        self._status_bar.showMessage(f"已打开：{path}")

    def _save_project(self) -> None:
        if not self._project or not self._store:
            QMessageBox.information(self, "提示", "当前无项目，请先新建或打开项目")
            return
        self._store.save_project(self._project)
        self._status_bar.showMessage("项目已保存")

    def _auto_save(self) -> None:
        """校对时自动保存修改行。"""
        if self._project and self._store:
            # 只更新有修改的行
            from app.models import ProofStatus
            for page in self._project.pages:
                for block in page.blocks:
                    for line in block.lines:
                        if line.id and line.proof_status in (
                            ProofStatus.MODIFIED, ProofStatus.OK
                        ):
                            self._store.update_line(line)

    # ── 工作流控制 ─────────────────────────────────────────────

    def _on_images_ready(self, paths: List[str]) -> None:
        """图片选择完成 → 构建 Page 对象 → 跳到版面分析步骤。"""
        if not self._project:
            # 自动建立临时项目（无保存路径）
            self._project = OcrProject(name="未命名项目")

        from PIL import Image
        pages = []
        for i, path in enumerate(paths):
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0
            page = Page(image_path=path, width=w, height=h, page_number=i + 1)
            pages.append(page)

        self._project.pages = pages
        self._layout_panel.set_pages(pages)
        self._step_bar.set_enabled_up_to(STEP_LAYOUT)
        self._go_to_step(STEP_LAYOUT)

    def _start_layout_analysis(self) -> None:
        if not self._project or not self._project.pages:
            return
        self._layout_panel.run_button.setEnabled(False)
        self._status_bar.showMessage("版面分析中…")
        self._layout_worker = LayoutWorker(self._project.pages)
        self._layout_worker.all_done.connect(self._on_layout_done)
        self._layout_worker.error.connect(self._on_worker_error)
        self._layout_worker.start()

    def _on_layout_done(self, pages: List[Page]) -> None:
        self._project.pages = pages
        self._layout_panel.show_analysis_result(pages)
        self._layout_panel.run_button.setEnabled(True)
        self._step_bar.set_enabled_up_to(STEP_OCR)
        self._status_bar.showMessage(f"版面分析完成：{len(pages)} 页")
        if self._store:
            self._store.save_project(self._project)

    def _start_ocr(self) -> None:
        if not self._project or not self._project.pages:
            return
        from app.core.ocr_runner import OcrWorker
        self._ocr_panel.set_pages(self._project.pages)
        self._go_to_step(STEP_OCR)
        self._ocr_worker = OcrWorker(self._project.pages)
        self._ocr_worker.page_done.connect(self._ocr_panel.on_progress)
        self._ocr_worker.all_done.connect(self._on_ocr_done)
        self._ocr_worker.error.connect(self._on_worker_error)
        self._ocr_worker.start()
        self._status_bar.showMessage("OCR 识别中…")

    def _on_ocr_done(self, pages: List[Page]) -> None:
        self._project.pages = pages
        flagged = self._proof_engine.auto_flag(pages)
        self._ocr_panel.on_recognition_complete(pages)
        self._hproof_panel.load_pages(pages)
        self._vproof_panel.load_pages(pages)
        self._step_bar.set_enabled_up_to(STEP_VPROOF)
        self._status_bar.showMessage(
            f"识别完成，自动标记 {flagged} 行低置信度内容"
        )
        if self._store:
            self._store.save_project(self._project)

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "错误", f"处理失败：\n{msg}")
        self._status_bar.showMessage("处理失败")

    def _show_export_dialog(self) -> None:
        if not self._project or not self._project.pages:
            QMessageBox.information(self, "提示", "请先完成 OCR 识别再导出")
            return
        dlg = ExportDialog(self._project, self)
        dlg.exec()

    # ── 辅助 ───────────────────────────────────────────────────

    def _prompt_project_name(self) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称：", text="新项目")
        return name.strip() or "新项目", ok

    def closeEvent(self, event) -> None:
        if self._store:
            self._store.close()
        super().closeEvent(event)

    def _show_ocr_settings(self) -> None:
        from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        from app.core.ocr_config import get_config
        dlg = ApiSettingsDialog(self)
        if dlg.exec():
            cfg = get_config()
            if cfg["mode"] == "api":
                self._status_bar.showMessage(f"OCR 引擎：API 模式（{cfg['api_url']}）")
            else:
                self._status_bar.showMessage("OCR 引擎：本地模型")
