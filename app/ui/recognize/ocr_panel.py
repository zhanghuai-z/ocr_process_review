"""OCR workspace projection over immutable application page views."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.application import BlockView, OcrLineView, OcrPageView, OcrWorkspaceView, PageView
from app.core.workflow_state import WorkflowProgressState


def _tree_payload(kind: str, *uids: str) -> tuple[str, ...]:
    return (kind, *uids)


class _PageViewDirectory(QListWidget):
    """Page directory projection that accepts only immutable page views."""

    page_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageDirectoryList")
        self.setMaximumWidth(236)
        self.setMinimumWidth(196)
        self.setSpacing(10)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self._page_uids: tuple[str, ...] = ()
        self._suppress_signal = False
        self.currentRowChanged.connect(self._on_row_changed)

    def set_pages(self, pages: Iterable[PageView]) -> None:
        values = tuple(pages)
        if any(not isinstance(page, PageView) for page in values):
            raise TypeError("OCR page directory requires PageView values")
        self._suppress_signal = True
        try:
            self.clear()
            self._page_uids = tuple(page.page_uid for page in values)
            for page in values:
                source = page.source_path or page.image_path
                filename = Path(source).name if source else ""
                item = QListWidgetItem(f"{page.page_number:02d}  {filename}")
                item.setData(Qt.ItemDataRole.UserRole, page.page_uid)
                item.setSizeHint(QSize(0, 44))
                self.addItem(item)
        finally:
            self._suppress_signal = False

    def set_current_uid(self, page_uid: str) -> None:
        try:
            index = self._page_uids.index(page_uid)
        except ValueError as exc:
            raise ValueError(f"page UID is not present in OCR directory: {page_uid!r}") from exc
        self._suppress_signal = True
        try:
            self.setCurrentRow(index)
        finally:
            self._suppress_signal = False

    def _on_row_changed(self, index: int) -> None:
        if self._suppress_signal or index < 0 or index >= len(self._page_uids):
            return
        self.page_selected.emit(self._page_uids[index])


class _PageImage(QWidget):
    """Small image projection that never receives a mutable domain object."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setText("暂无页面图像")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self._image)

    def set_page(self, page: PageView | None) -> None:
        if page is None:
            self._image.clear()
            self._image.setText("暂无页面图像")
            return
        path = page.thumbnail_path or page.image_path or page.source_path
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            self._image.clear()
            self._image.setText(f"无法读取图像：{path}")
            return
        self._image.setText("")
        self._image.setPixmap(pixmap)
        self._image.setScaledContents(False)


class OcrPanel(QWidget):
    """Browse immutable page and layout views used by the OCR workflow."""

    page_selected = Signal(str)
    ocr_requested = Signal()
    go_to_proof_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workspace: OcrWorkspaceView | None = None
        self._pages: tuple[OcrPageView, ...] = ()
        self._pages_by_uid: dict[str, PageView] = {}
        self._current_page_uid = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.setContentsMargins(12, 12, 12, 4)
        title = QLabel("OCR 工作区")
        title.setObjectName("pageTitle")
        top.addWidget(title)
        top.addStretch()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(200)
        self._progress.setFixedHeight(16)
        self._progress.setFormat("%p%")
        self._progress.hide()
        top.addWidget(self._progress)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._directory = _PageViewDirectory()
        self._directory.page_selected.connect(self._on_page_selected)
        splitter.addWidget(self._directory)

        self._image = _PageImage()
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(self._image)
        splitter.addWidget(image_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 4, 0)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["内容", "置信度", "状态"])
        self._tree.setColumnWidth(0, 280)
        self._tree.setColumnWidth(1, 80)
        self._tree.currentItemChanged.connect(self._on_item_selected)
        right_layout.addWidget(self._tree)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(12, 8, 12, 8)
        self._status = QLabel("等待导入")
        self._status.setObjectName("muted")
        bottom.addWidget(self._status)
        bottom.addStretch()
        self._start_ocr = QPushButton("开始 OCR")
        self._start_ocr.setObjectName("primaryBtn")
        self._start_ocr.setEnabled(False)
        self._start_ocr.clicked.connect(self.ocr_requested.emit)
        bottom.addWidget(self._start_ocr)
        self._go_proof = QPushButton("进入校对")
        self._go_proof.setObjectName("secondaryBtn")
        self._go_proof.setEnabled(False)
        self._go_proof.clicked.connect(self.go_to_proof_requested.emit)
        bottom.addWidget(self._go_proof)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------ public API

    def set_workspace(self, workspace: OcrWorkspaceView) -> None:
        if not isinstance(workspace, OcrWorkspaceView):
            raise TypeError("OCR panel requires OcrWorkspaceView")
        self._workspace = workspace
        self._pages = workspace.pages
        page_values = tuple(item.page for item in workspace.pages)
        self._pages_by_uid = {page.page_uid: page for page in page_values}
        self._directory.set_pages(page_values)
        self._tree.clear()
        target_uid = self._current_page_uid if self._current_page_uid in self._pages_by_uid else (
            page_values[0].page_uid if page_values else ""
        )
        self._set_current_uid(target_uid)
        self._populate_tree()
        has_ocr = any(page.has_current_ocr for page in workspace.pages)
        has_layout = any(page.page.layout_revision is not None for page in workspace.pages)
        self._start_ocr.setEnabled(bool(page_values) and has_layout and not all(
            page.has_current_ocr for page in workspace.pages
        ))
        self._go_proof.setEnabled(has_ocr)
        if not page_values:
            self._status.setText("等待导入")
        elif has_ocr:
            self._status.setText(
                f"已载入 {len(page_values)} 页，{workspace.line_count} 行 OCR 观察"
            )
        elif has_layout:
            self._status.setText("已载入页面版面，等待 OCR")
        else:
            self._status.setText("已导入页面，等待版面分析")

    def on_progress(self, progress: WorkflowProgressState) -> None:
        if not isinstance(progress, WorkflowProgressState):
            raise TypeError("OCR panel requires WorkflowProgressState")
        total_pages = max(1, progress.total_pages)
        completed = max(0, min(progress.completed_pages, total_pages))
        value = int(round(completed / total_pages * 100))
        if progress.total > 0:
            value = max(value, int(round(progress.current / progress.total * 100)))
        self._progress.show()
        self._progress.setValue(max(0, min(100, value)))
        self._status.setText(progress.message or "OCR 处理中")
        self._start_ocr.setEnabled(False)

    def set_current_page_uid(self, page_uid: str) -> None:
        if page_uid not in self._pages_by_uid:
            raise ValueError(f"page UID is not present in OCR panel: {page_uid!r}")
        self._set_current_uid(page_uid)
        self._directory.set_current_uid(page_uid)

    def set_ocr_enabled(self, enabled: bool) -> None:
        self._start_ocr.setEnabled(bool(enabled) and bool(self._pages))

    def finish_progress(self, message: str = "") -> None:
        self._progress.hide()
        if message:
            self._status.setText(message)

    def reset(self) -> None:
        self._workspace = None
        self._pages = ()
        self._pages_by_uid = {}
        self._current_page_uid = ""
        self._directory.set_pages(())
        self._tree.clear()
        self._image.set_page(None)
        self._progress.hide()
        self._status.setText("等待导入")
        self._start_ocr.setEnabled(False)
        self._go_proof.setEnabled(False)

    # ------------------------------------------------------------------ private

    def _set_current_uid(self, page_uid: str) -> None:
        self._current_page_uid = page_uid
        self._image.set_page(self._pages_by_uid.get(page_uid))

    def _on_page_selected(self, page_uid: str) -> None:
        if page_uid not in self._pages_by_uid:
            return
        self._set_current_uid(page_uid)
        self.page_selected.emit(page_uid)

    def _populate_tree(self) -> None:
        self._tree.clear()
        for page_view in self._pages:
            page = page_view.page
            page_item = QTreeWidgetItem(self._tree, [f"第 {page.page_number} 页", "", ""])
            page_item.setData(0, Qt.ItemDataRole.UserRole, _tree_payload("page", page.page_uid))
            regions_by_block = {
                block.block_uid: tuple(
                    region for region in page_view.regions if region.block_uid == block.block_uid
                )
                for block in page.blocks
            }
            for block in page.blocks:
                lines = tuple(
                    line
                    for region in regions_by_block[block.block_uid]
                    for line in region.lines
                )
                block_item = QTreeWidgetItem(
                    page_item,
                    [
                        f"[{self._block_label(block)}]",
                        self._average_confidence(lines),
                        f"{len(lines)} 行" if page_view.has_current_ocr else "版面",
                    ],
                )
                block_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    _tree_payload("block", page.page_uid, block.block_uid),
                )
                for line in lines:
                    line_item = QTreeWidgetItem(
                        block_item,
                        [line.text[:48], f"{line.confidence:.2f}", "观察"],
                    )
                    line_item.setData(
                        0,
                        Qt.ItemDataRole.UserRole,
                        _tree_payload("line", page.page_uid, block.block_uid, line.line_uid),
                    )
            unbound_regions = tuple(region for region in page_view.regions if region.block_uid is None)
            if unbound_regions:
                unbound_item = QTreeWidgetItem(page_item, ["[未绑定 OCR 区域]", "--", "异常"])
                for region in unbound_regions:
                    for line in region.lines:
                        QTreeWidgetItem(
                            unbound_item,
                            [line.text[:48], f"{line.confidence:.2f}", "观察"],
                        )
            page_item.setExpanded(True)

    @staticmethod
    def _block_label(block: BlockView) -> str:
        return block.source_label or block.block_type.value

    @staticmethod
    def _average_confidence(lines: tuple[OcrLineView, ...]) -> str:
        if not lines:
            return "--"
        return f"{sum(line.confidence for line in lines) / len(lines):.2f}"

    def _on_item_selected(self, current: QTreeWidgetItem | None, _previous) -> None:
        if current is None:
            return
        payload = current.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(payload, tuple) or len(payload) < 2:
            return
        page_uid = payload[1]
        if page_uid in self._pages_by_uid:
            self._set_current_uid(page_uid)
        if payload[0] == "block" and len(payload) == 3:
            self._status.setText(f"已选中版面块：{payload[2]}")


__all__ = ["OcrPanel"]
