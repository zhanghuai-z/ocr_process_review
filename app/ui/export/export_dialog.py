"""导出对话框：多格式选择 + 路径 + 导出进度。"""
from __future__ import annotations
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout,
)

from app.models import OcrProject


class ExportWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, project: OcrProject, formats: List[str], out_dir: str):
        super().__init__()
        self._project = project
        self._formats = formats
        self._out_dir = out_dir

    def run(self) -> None:
        from app.export import get_exporter
        errors = []
        for fmt in self._formats:
            self.progress.emit(f"正在导出 {fmt.upper()}…")
            try:
                exporter = get_exporter(fmt)
                out_path = str(Path(self._out_dir) / f"{self._project.name}.{fmt}")
                exporter.export(self._project, out_path)
            except Exception as e:
                errors.append(f"{fmt}: {e}")
        if errors:
            self.finished.emit(False, "\n".join(errors))
        else:
            self.finished.emit(True, "")


class ExportDialog(QDialog):
    """导出格式选择对话框。"""

    def __init__(self, project: OcrProject, parent=None):
        super().__init__(parent)
        self._project = project
        self._worker: ExportWorker | None = None
        self.setWindowTitle("导出")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 格式选择
        fmt_group = QGroupBox("选择导出格式")
        fmt_layout = QVBoxLayout(fmt_group)
        self._checkboxes: dict[str, QCheckBox] = {}
        formats = [
            ("txt",  "纯文本 (.txt)"),
            ("rtf",  "富文本 (.rtf)"),
            ("pdf",  "PDF (.pdf)"),
            ("xml",  "XML (.xml)"),
            ("html", "HTML (.html)"),
            ("docx", "Word 文档 (.docx)"),
        ]
        for fmt, label in formats:
            cb = QCheckBox(label)
            cb.setChecked(fmt in ("txt", "xml"))
            self._checkboxes[fmt] = cb
            fmt_layout.addWidget(cb)
        layout.addWidget(fmt_group)

        # 输出目录
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("输出目录："))
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("选择保存目录…")
        dir_layout.addWidget(self._dir_edit)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        layout.addLayout(dir_layout)

        # 进度
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color:#aaa;")
        layout.addWidget(self._progress_lbl)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # 按钮
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("开始导出")
        btns.accepted.connect(self._start_export)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._dir_edit.setText(d)

    def _start_export(self) -> None:
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return

        selected = [fmt for fmt, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一种导出格式")
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # 不确定模式

        self._worker = ExportWorker(self._project, selected, out_dir)
        self._worker.progress.connect(self._progress_lbl.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, ok: bool, msg: str) -> None:
        self._progress_bar.setVisible(False)
        if ok:
            self._progress_lbl.setText("✓ 导出完成")
            QMessageBox.information(self, "完成", "所有格式导出成功！")
            self.accept()
        else:
            QMessageBox.critical(self, "导出失败", msg)
