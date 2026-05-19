"""导出对话框：多格式选择 + 路径 + 导出进度。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout,
)

from app.models import OcrProject
from app.services.export_service import build_export_path


@dataclass(frozen=True)
class ExportFormatResult:
    fmt: str
    out_path: str
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class ExportRunResult:
    results: list[ExportFormatResult]

    @property
    def successes(self) -> list[ExportFormatResult]:
        return [result for result in self.results if result.ok]

    @property
    def failures(self) -> list[ExportFormatResult]:
        return [result for result in self.results if not result.ok]

    @property
    def all_ok(self) -> bool:
        return bool(self.results) and not self.failures

    @property
    def any_success(self) -> bool:
        return bool(self.successes)

    def summary(self) -> str:
        lines: list[str] = []
        if self.successes:
            lines.append("已导出：")
            lines.extend(f"  • {r.fmt.upper()}: {r.out_path}" for r in self.successes)
        if self.failures:
            lines.append("失败：")
            lines.extend(f"  • {r.fmt.upper()}: {r.error}" for r in self.failures)
        return "\n".join(lines)


class ExportWorker(QThread):
    progress = Signal(str, int, int)
    completed = Signal(object)

    def __init__(self, project: OcrProject, formats: List[str], out_dir: str):
        super().__init__()
        self._project = project
        self._formats = formats
        self._out_dir = out_dir

    def run(self) -> None:
        from app.export import get_exporter
        results: list[ExportFormatResult] = []
        total = len(self._formats)
        for index, fmt in enumerate(self._formats, start=1):
            self.progress.emit(f"正在导出 {fmt.upper()}…", index - 1, total)
            out_path = str(build_export_path(self._out_dir, self._project.name, fmt))
            try:
                exporter = get_exporter(fmt)
                exporter.export(self._project, out_path)
                results.append(ExportFormatResult(fmt=fmt, out_path=out_path, ok=True))
                self.progress.emit(f"{fmt.upper()} 导出完成", index, total)
            except Exception as e:
                results.append(ExportFormatResult(fmt=fmt, out_path=out_path, ok=False, error=str(e)))
                self.progress.emit(f"{fmt.upper()} 导出失败", index, total)
        self.completed.emit(ExportRunResult(results))


class ExportDialog(QDialog):
    """导出格式选择对话框。"""

    def __init__(self, project: OcrProject, parent=None):
        super().__init__(parent)
        self._project = project
        self._worker: ExportWorker | None = None
        self._btn_start = None
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
            ("md",   "Markdown (.md)"),
            ("rtf",  "富文本 (.rtf)"),
            ("pdf",  "PDF (.pdf)"),
            ("xml",  "XML (.xml)"),
            ("html", "HTML (.html)"),
            ("docx", "Word 文档 (.docx)"),
        ]
        for fmt, label in formats:
            cb = QCheckBox(label)
            cb.setChecked(fmt in ("txt", "md", "xml"))
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
        self._btn_start = btns.button(QDialogButtonBox.StandardButton.Ok)
        self._btn_start.setText("开始导出")
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
        self._progress_bar.setRange(0, len(selected))
        self._progress_bar.setValue(0)
        self._progress_lbl.setText("准备导出…")
        if self._btn_start is not None:
            self._btn_start.setEnabled(False)

        self._worker = ExportWorker(self._project, selected, out_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, message: str, current: int, total: int) -> None:
        self._progress_lbl.setText(message)
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(max(0, min(current, total)))

    def _on_finished(self, result: ExportRunResult) -> None:
        if result.all_ok:
            self._progress_bar.setValue(self._progress_bar.maximum())
            self._progress_lbl.setText("✓ 导出完成")
            self.accept()
            return
        if self._btn_start is not None:
            self._btn_start.setEnabled(True)
        if result.any_success:
            self._progress_bar.setValue(self._progress_bar.maximum())
            self._progress_lbl.setText("部分导出完成")
            QMessageBox.warning(self, "部分导出完成", result.summary())
            return
        self._progress_bar.setVisible(False)
        self._progress_lbl.setText("导出失败")
        QMessageBox.critical(self, "导出失败", result.summary())
