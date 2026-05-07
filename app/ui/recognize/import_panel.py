"""图片导入面板：支持拖拽和浏览选择图片/PDF。"""
from __future__ import annotations
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


class DropArea(QLabel):
    """可拖拽放入文件的区域。"""
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("将图片或 PDF 拖拽至此\n或点击下方按钮选择文件")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setObjectName("dropArea")
        self.setStyleSheet(
            "QLabel#dropArea{border:2px dashed #b8d4ff; border-radius:8px;"
            "background:#fafcff; color:#1a73e8; font-size:14px;}"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.files_dropped.emit(paths)


_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}


class ImportPanel(QWidget):
    """
    步骤1: 文件导入面板。
    发出 images_ready(List[str]) 信号，payload 为展开后的图片路径列表（PDF 已转图片）。
    """
    images_ready = Signal(list)   # List[str] — 图片文件绝对路径

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: List[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("① 导入文件")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        # 拖拽区
        self._drop_area = DropArea()
        self._drop_area.files_dropped.connect(self._add_paths)
        layout.addWidget(self._drop_area)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_add = QPushButton("选择文件…")
        btn_add.clicked.connect(self._browse)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 文件列表
        list_label = QLabel("已选文件")
        list_label.setObjectName("noteLabel")
        layout.addWidget(list_label)

        self._list = QListWidget()
        self._list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._list)

        # 下一步按钮
        self._btn_next = QPushButton("开始版面分析 →")
        self._btn_next.setEnabled(False)
        self._btn_next.setObjectName("primaryBtn"); self._btn_next.setMinimumHeight(34)
        self._btn_next.clicked.connect(self._emit_ready)
        layout.addWidget(self._btn_next, alignment=Qt.AlignmentFlag.AlignRight)

    # ------------------------------------------------------------------ slots

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片或 PDF",
            "",
            "图片/PDF (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.pdf)",
        )
        if paths:
            self._add_paths(paths)

    def _add_paths(self, paths: List[str]) -> None:
        for p in paths:
            if Path(p).suffix.lower() in _SUPPORTED_EXT and p not in self._paths:
                self._paths.append(p)
                self._list.addItem(QListWidgetItem(Path(p).name + f"  [{p}]"))
        self._btn_next.setEnabled(bool(self._paths))

    def _clear(self) -> None:
        self._paths.clear()
        self._list.clear()
        self._btn_next.setEnabled(False)

    def _emit_ready(self) -> None:
        self.images_ready.emit(list(self._paths))

    def reset(self) -> None:
        """清空导入面板（新建项目时调用）。"""
        self._clear()
