"""OCR workspace projection over immutable export snapshots."""
from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.workflow_state import WorkflowProgressState
from app.models.export_snapshot import ExportPageSnapshot, ExportProjectSnapshot
from app.models.layout_snapshot import LayoutBlockSnapshot
from app.models.ocr_records import OcrLine
from app.models.project_session import PageRecord
from app.ui.widgets.page_directory import PageDirectoryList


def _tree_payload(kind: str, *uids: str) -> tuple[str, ...]:
    return (kind, *uids)


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

    def set_page(self, page: PageRecord | None) -> None:
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
    """Browse adopted layout and OCR observations from one export snapshot."""

    page_selected = Signal(str)
    ocr_requested = Signal()
    go_to_proof_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: tuple[PageRecord, ...] = ()
        self._snapshots: tuple[ExportPageSnapshot, ...] = ()
        self._pages_by_uid: dict[str, PageRecord] = {}
        self._snapshots_by_uid: dict[str, ExportPageSnapshot] = {}
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
        self._directory = PageDirectoryList()
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

    def set_pages(self, pages: Iterable[PageRecord]) -> None:
        records = tuple(pages)
        if any(not isinstance(page, PageRecord) for page in records):
            raise TypeError("OCR panel requires PageRecord values")
        self._pages = records
        self._pages_by_uid = {page.uid: page for page in records}
        self._snapshots = ()
        self._snapshots_by_uid = {}
        self._directory.set_pages(records)
        self._tree.clear()
        self._set_current_uid(records[0].uid if records else "")
        self._start_ocr.setEnabled(False)
        self._go_proof.setEnabled(False)
        self._status.setText("已导入页面，等待版面分析") if records else self._status.setText("等待导入")

    def set_snapshot(self, snapshot: ExportProjectSnapshot) -> None:
        if not isinstance(snapshot, ExportProjectSnapshot):
            raise TypeError("OCR panel requires ExportProjectSnapshot")
        self._snapshots = tuple(snapshot.pages)
        self._snapshots_by_uid = {
            page.page.uid: page for page in self._snapshots
        }
        self._pages = tuple(page.page for page in self._snapshots)
        self._pages_by_uid = {page.uid: page for page in self._pages}
        self._directory.set_pages(self._pages)
        target_uid = self._current_page_uid if self._current_page_uid in self._pages_by_uid else (
            self._pages[0].uid if self._pages else ""
        )
        self._set_current_uid(target_uid)
        self._populate_tree()
        has_ocr = any(page.active_ocr_batch is not None for page in self._snapshots)
        self._start_ocr.setEnabled(bool(self._pages) and not has_ocr)
        self._go_proof.setEnabled(has_ocr)
        self._status.setText(
            f"已载入 {len(self._pages)} 页会话快照，{sum(len(page.ocr_lines) for page in self._snapshots)} 行 OCR 观察"
        )

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
        self._pages = ()
        self._snapshots = ()
        self._pages_by_uid = {}
        self._snapshots_by_uid = {}
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
        for page_snapshot in self._snapshots:
            page = page_snapshot.page
            page_item = QTreeWidgetItem(self._tree, [f"第 {page.page_number} 页", "", ""])
            page_item.setData(0, Qt.ItemDataRole.UserRole, _tree_payload("page", page.uid))
            for block in page_snapshot.layout.blocks:
                lines = self._lines_for_block(page_snapshot, block)
                block_item = QTreeWidgetItem(
                    page_item,
                    [
                        f"[{block.source_label or block.block_type.value}]",
                        self._average_confidence(lines),
                        f"{len(lines)} 行",
                    ],
                )
                block_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    _tree_payload("block", page.uid, block.uid),
                )
                for line in lines:
                    line_item = QTreeWidgetItem(
                        block_item,
                        [line.text[:48], f"{line.confidence:.2f}", "观察"],
                    )
                    line_item.setData(
                        0,
                        Qt.ItemDataRole.UserRole,
                        _tree_payload("line", page.uid, block.uid, line.uid),
                    )
            page_item.setExpanded(True)

    def _lines_for_block(
        self,
        page_snapshot: ExportPageSnapshot,
        block: LayoutBlockSnapshot,
    ) -> tuple[OcrLine, ...]:
        region_uids = {
            binding.target_uid
            for binding in page_snapshot.bindings
            if binding.source_uid == block.uid
        }
        return tuple(sorted(
            (
                line
                for line in page_snapshot.ocr_lines
                if line.region_uid in region_uids
            ),
            key=lambda line: (line.order, line.uid),
        ))

    @staticmethod
    def _average_confidence(lines: tuple[OcrLine, ...]) -> str:
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
        if payload[0] == "line" and len(payload) == 4:
            self._status.setText(f"已选中 OCR 行：{payload[3]}")
        elif payload[0] == "block" and len(payload) == 3:
            self._status.setText(f"已选中版面块：{payload[2]}")


__all__ = ["OcrPanel"]
