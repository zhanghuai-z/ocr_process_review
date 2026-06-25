"""图片导入面板：支持拖拽、点击导入和待处理文件队列。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from app.ui.widgets.effects import apply_soft_shadow
from app.utils.icon_manager import get_icon


_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}


def _file_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "PDF"
    if suffix in {".tif", ".tiff"}:
        return "TIFF"
    return "IMAGE"


class DropArea(QFrame):
    """可拖拽/点击放入文件的区域。"""

    files_dropped = Signal(list)
    browse_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(210)
        self.setObjectName("importDropArea")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel()
        icon.setObjectName("importDropIcon")
        icon.setPixmap(get_icon("export", color="#6B6B6B", size=28).pixmap(QSize(28, 28)))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignCenter)

        title = QLabel("拖拽文件至此，或点击导入")
        title.setObjectName("importDropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("支持 PDF、JPG、PNG、BMP、TIFF。导入文件即开始当前项目。")
        desc.setObjectName("importDropDesc")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.browse_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.files_dropped.emit(paths)


class _FileRowWidget(QFrame):
    """导入队列中的单行文件信息。"""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("importFileRow")
        self.setMinimumHeight(42)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(10)

        icon_name = "folder" if _file_kind(path) == "PDF" else "doc_image"
        icon = QLabel()
        icon.setObjectName("importFileIcon")
        icon.setPixmap(get_icon(icon_name).pixmap(QSize(18, 18)))
        icon.setFixedSize(22, 22)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(icon)

        name_box = QVBoxLayout()
        name_box.setContentsMargins(0, 0, 0, 0)
        name_box.setSpacing(1)
        name = QLabel(Path(path).name)
        name.setObjectName("importFileName")
        name.setToolTip(path)
        source = QLabel(str(Path(path).parent))
        source.setObjectName("importFilePath")
        source.setToolTip(path)
        name_box.addWidget(name)
        name_box.addWidget(source)
        row.addLayout(name_box, 1)

        kind = QLabel(_file_kind(path))
        kind.setObjectName("importKindPill")
        kind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(kind)

        status = QLabel("就绪")
        status.setObjectName("importReadyPill")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(status)


class ImportPanel(QWidget):
    """
    文件导入面板。

    发出 images_ready(List[str]) 信号，payload 为展开前的文件路径列表；
    PDF 展开仍由主窗口 ImportWorker / ImportService 负责。
    """

    images_ready = Signal(list)
    open_project_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("importRoot")
        self._paths: List[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(3)
        title = QLabel("项目入口")
        title.setObjectName("importTitle")
        title_box.addWidget(title)
        subtitle = QLabel("导入图片或 PDF 后，即进入当前项目工作流。")
        subtitle.setObjectName("importSubtitle")
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)

        btn_add = QPushButton("打开项目")
        btn_add.setIcon(get_icon("folder", color="#FFFFFF", stroke_width=2.0))
        btn_add.setIconSize(QSize(16, 16))
        btn_add.setObjectName("darkBtn")
        btn_add.clicked.connect(self.open_project_requested)
        header.addWidget(btn_add)
        layout.addLayout(header)

        workspace = QHBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(14)
        layout.addLayout(workspace, 1)

        left_card = QFrame()
        left_card.setObjectName("importCard")
        apply_soft_shadow(left_card, blur_radius=20, y_offset=5, alpha=12)
        left = QVBoxLayout(left_card)
        left.setContentsMargins(18, 18, 18, 18)
        left.setSpacing(14)

        section = QLabel("文件导入")
        section.setObjectName("importSectionTitle")
        left.addWidget(section)

        self._drop_area = DropArea()
        self._drop_area.files_dropped.connect(self._add_paths)
        self._drop_area.browse_requested.connect(self._browse)
        left.addWidget(self._drop_area, 1)

        hint = QLabel("导入后会先渲染页面图片，再进入版面分析。关闭项目会提示保存当前项目。")
        hint.setObjectName("importHint")
        hint.setWordWrap(True)
        left.addWidget(hint)
        workspace.addWidget(left_card, 5)

        right_card = QFrame()
        right_card.setObjectName("importCard")
        apply_soft_shadow(right_card, blur_radius=20, y_offset=5, alpha=12)
        right = QVBoxLayout(right_card)
        right.setContentsMargins(16, 16, 16, 16)
        right.setSpacing(10)

        queue_header = QHBoxLayout()
        queue_header.setContentsMargins(0, 0, 0, 0)
        queue_title = QLabel("待处理队列")
        queue_title.setObjectName("importSectionTitle")
        queue_header.addWidget(queue_title)
        queue_header.addStretch()
        self._count_pill = QLabel("0 个文件")
        self._count_pill.setObjectName("importCountPill")
        self._count_pill.setFixedHeight(28)
        self._count_pill.setMinimumWidth(86)
        self._count_pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._count_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        queue_header.addWidget(self._count_pill)
        right.addLayout(queue_header)

        self._empty_label = QLabel("队列为空。拖拽文件到左侧，或点击导入文件。")
        self._empty_label.setObjectName("importEmpty")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        right.addWidget(self._empty_label)

        self._list = QListWidget()
        self._list.setObjectName("importQueueList")
        self._list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right.addWidget(self._list, 1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        self._btn_clear = QPushButton("清空列表")
        self._btn_clear.setIcon(get_icon("delete", color="#6B6B6B"))
        self._btn_clear.setIconSize(QSize(16, 16))
        self._btn_clear.setObjectName("ghostBtn")
        self._btn_clear.clicked.connect(self._clear)
        action_row.addWidget(self._btn_clear)
        action_row.addStretch()
        self._btn_next = QPushButton("开始版面分析")
        self._btn_next.setObjectName("primaryBtn")
        self._btn_next.setMinimumHeight(34)
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self._emit_ready)
        action_row.addWidget(self._btn_next)
        right.addLayout(action_row)
        workspace.addWidget(right_card, 4)

        self._refresh_queue_state()

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
                item = QListWidgetItem()
                row = _FileRowWidget(p)
                item.setSizeHint(row.sizeHint())
                self._list.addItem(item)
                self._list.setItemWidget(item, row)
        self._refresh_queue_state()

    def _clear(self) -> None:
        self._paths.clear()
        self._list.clear()
        self._refresh_queue_state()

    def _refresh_queue_state(self) -> None:
        count = len(self._paths)
        self._count_pill.setText(f"{count} 个文件")
        self._empty_label.setVisible(count == 0)
        self._list.setVisible(count > 0)
        self._btn_clear.setEnabled(count > 0)
        self._btn_next.setEnabled(count > 0)

    def _emit_ready(self) -> None:
        self.images_ready.emit(list(self._paths))

    def reset(self) -> None:
        """清空导入面板（关闭项目或重新导入时调用）。"""
        self._clear()
