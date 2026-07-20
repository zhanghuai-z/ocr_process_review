"""Export dialog over one immutable captured project snapshot."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from app.export import get_exporter
from app.models.export_snapshot import ExportProjectSnapshot
from app.services.export_service import build_export_path
from app.ui.widgets.effects import apply_soft_shadow


@dataclass(frozen=True, slots=True)
class ExportFormatResult:
    fmt: str
    out_path: str
    ok: bool
    error: str = ""


@dataclass(frozen=True, slots=True)
class ExportRunResult:
    results: tuple[ExportFormatResult, ...]

    @property
    def successes(self) -> tuple[ExportFormatResult, ...]:
        return tuple(result for result in self.results if result.ok)

    @property
    def failures(self) -> tuple[ExportFormatResult, ...]:
        return tuple(result for result in self.results if not result.ok)

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
            lines.extend(
                f"  • {result.fmt.upper()}: {result.out_path}"
                for result in self.successes
            )
        if self.failures:
            lines.append("失败：")
            lines.extend(
                f"  • {result.fmt.upper()}: {result.error}"
                for result in self.failures
            )
        return "\n".join(lines)


class ExportDialog(QDialog):
    """Choose formats and export the supplied transaction snapshot."""

    def __init__(self, snapshot: ExportProjectSnapshot, parent=None) -> None:
        super().__init__(parent)
        if not isinstance(snapshot, ExportProjectSnapshot):
            raise TypeError("ExportDialog requires ExportProjectSnapshot")
        self._snapshot = snapshot
        self._btn_start: QPushButton | None = None
        self.setObjectName("exportDialog")
        self.setWindowTitle("导出")
        self.setMinimumWidth(560)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        card = QFrame()
        card.setObjectName("exportDialogCard")
        apply_soft_shadow(card, blur_radius=20, y_offset=4, alpha=14)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(14)
        layout.addWidget(card)

        title = QLabel("导出文档")
        title.setObjectName("dialogTitle")
        card_layout.addWidget(title)
        subtitle = QLabel("导出使用已捕获的会话快照，不读取运行中的项目状态。")
        subtitle.setObjectName("dialogSubtitle")
        card_layout.addWidget(subtitle)

        fmt_group = QGroupBox("选择导出格式")
        fmt_layout = QGridLayout(fmt_group)
        fmt_layout.setHorizontalSpacing(18)
        fmt_layout.setVerticalSpacing(4)
        self._checkboxes: dict[str, QCheckBox] = {}
        formats = (
            ("txt", "纯文本 (.txt)"),
            ("json", "Export IR JSON (.json)"),
            ("md", "Markdown (.md)"),
            ("rtf", "富文本 (.rtf)"),
            ("pdf-single", "PDF 原图单层 (.pdf)"),
            ("pdf-dual", "PDF 原图+可搜索文本 (.pdf)"),
            ("xml", "XML (.xml)"),
            ("html", "HTML (.html)"),
            ("docx", "Word 文档 (.docx)"),
        )
        for index, (fmt, label) in enumerate(formats):
            checkbox = QCheckBox(label)
            checkbox.setChecked(fmt in {"txt", "json", "md", "xml"})
            self._checkboxes[fmt] = checkbox
            fmt_layout.addWidget(checkbox, index // 2, index % 2)
        card_layout.addWidget(fmt_group)

        directory_layout = QHBoxLayout()
        directory_layout.addWidget(QLabel("输出目录："))
        self._directory_edit = QLineEdit()
        self._directory_edit.setPlaceholderText("选择保存目录…")
        directory_layout.addWidget(self._directory_edit)
        browse = QPushButton("浏览…")
        browse.setObjectName("secondaryBtn")
        browse.clicked.connect(self._browse_dir)
        directory_layout.addWidget(browse)
        card_layout.addLayout(directory_layout)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("muted")
        card_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        card_layout.addWidget(self._progress_bar)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btn_start = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._btn_start.setText("开始导出")
        self._btn_start.setObjectName("primaryBtn")
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel is not None:
            cancel.setText("取消")
            cancel.setObjectName("secondaryBtn")
        buttons.accepted.connect(self._start_export)
        buttons.rejected.connect(self.reject)
        card_layout.addWidget(buttons)

    def _browse_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self._directory_edit.setText(directory)

    def _start_export(self) -> None:
        out_dir = self._directory_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        selected = tuple(fmt for fmt, checkbox in self._checkboxes.items() if checkbox.isChecked())
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一种导出格式")
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(selected))
        self._progress_bar.setValue(0)
        if self._btn_start is not None:
            self._btn_start.setEnabled(False)

        results: list[ExportFormatResult] = []
        for index, fmt in enumerate(selected, start=1):
            self._progress_label.setText(f"正在导出 {fmt.upper()}…")
            QApplication.processEvents()
            output_path = ""
            try:
                output_path = str(
                    build_export_path(out_dir, self._snapshot.project.name, fmt)
                )
                get_exporter(fmt).export(self._snapshot, output_path)
                result = ExportFormatResult(fmt, output_path, True)
                self._progress_label.setText(f"{fmt.upper()} 导出完成")
            except Exception as exc:
                result = ExportFormatResult(fmt, output_path, False, str(exc))
                self._progress_label.setText(f"{fmt.upper()} 导出失败")
            results.append(result)
            self._progress_bar.setValue(index)
            QApplication.processEvents()

        result = ExportRunResult(tuple(results))
        if result.all_ok:
            self._progress_label.setText("导出完成")
            self.accept()
            return
        if self._btn_start is not None:
            self._btn_start.setEnabled(True)
        if result.any_success:
            QMessageBox.warning(self, "部分导出完成", result.summary())
            return
        self._progress_bar.setVisible(False)
        self._progress_label.setText("导出失败")
        QMessageBox.critical(self, "导出失败", result.summary())


__all__ = ["ExportDialog", "ExportFormatResult", "ExportRunResult"]
