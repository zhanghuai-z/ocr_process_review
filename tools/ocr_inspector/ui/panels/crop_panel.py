"""OCR Inspector — 文本找图 & 切图面板.

两个功能:
1. 文本查找 (TextMatch tab):
   输入文本 → 搜索当前文档所有 chars / lines → 显示匹配列表 → 点击 → 预览切图
   每个匹配项显示: 匹配文本 | kind(char/line) | bbox | bbox_source
   右下角"保存选中切图"按钮

2. 手动切图 (Manual tab):
   输入 x, y, w, h 坐标 → 切图预览 → 保存
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from tools.ocr_inspector.crop import (
    TextMatch, crop_bbox, find_text_matches, pil_to_qpixmap, save_crop,
)
from tools.ocr_inspector.models.ir import BBox
from tools.ocr_inspector.state import AppState


# ---------------------------------------------------------------------------
# PIL → QPixmap in background thread
# ---------------------------------------------------------------------------

class _CropWorker(QThread):
    """Background thread: runs crop_bbox, emits QPixmap."""
    done = Signal(object)    # QPixmap or None
    error = Signal(str)

    def __init__(self, image_path: str, bbox: BBox, padding: int = 4):
        super().__init__()
        self._path = image_path
        self._bbox = bbox
        self._padding = padding

    def run(self):
        try:
            pil_img = crop_bbox(self._path, self._bbox, padding=self._padding)
            pix = pil_to_qpixmap(pil_img)
            self.done.emit(pix)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Crop preview widget
# ---------------------------------------------------------------------------

class _CropPreview(QWidget):
    """Shows a scrollable crop preview with 'No image' placeholder."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("（未选择）")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #888; font-size: 11px;")
        self._label.setWordWrap(True)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(140)
        layout.addWidget(self._scroll)

        self._info = QLabel("")
        self._info.setStyleSheet("color: #555; font-size: 10px;")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

    def show_loading(self):
        self._label.setText("加载中…")
        self._label.setPixmap(QPixmap())
        self._info.setText("")

    def show_pixmap(self, pix: QPixmap, info: str = ""):
        scaled = pix.scaled(
            self._scroll.viewport().width() - 8,
            max(120, self._scroll.viewport().height() - 8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._info.setText(info)

    def show_error(self, msg: str):
        self._label.setText(f"[错误] {msg}")
        self._label.setPixmap(QPixmap())
        self._info.setText("")

    def clear(self):
        self._label.setText("（未选择）")
        self._label.setPixmap(QPixmap())
        self._info.setText("")


# ---------------------------------------------------------------------------
# Text-search tab
# ---------------------------------------------------------------------------

class _TextSearchTab(QWidget):
    """文本查找 tab: 输入文本 → 搜索 → 列表 → 预览切图."""

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        self._matches: List[TextMatch] = []
        self._worker: Optional[_CropWorker] = None

        layout = QVBoxLayout(self)

        # Search bar
        search_row = QHBoxLayout()
        self._query = QLineEdit()
        self._query.setPlaceholderText("输入要查找的文字…")
        self._query.returnPressed.connect(self._do_search)
        search_row.addWidget(self._query)

        self._btn_search = QPushButton("查找")
        self._btn_search.clicked.connect(self._do_search)
        search_row.addWidget(self._btn_search)
        layout.addLayout(search_row)

        # Options
        opts_row = QHBoxLayout()
        self._chk_chars = QCheckBox("字/词框")
        self._chk_chars.setChecked(True)
        self._chk_chars.setToolTip("搜索字级 bbox（仅 bbox_source=ocr 的真实字框）")
        opts_row.addWidget(self._chk_chars)
        self._chk_lines = QCheckBox("行框")
        self._chk_lines.setChecked(True)
        self._chk_lines.setToolTip("搜索行级 bbox")
        opts_row.addWidget(self._chk_lines)
        self._chk_case = QCheckBox("区分大小写")
        self._chk_case.setChecked(False)
        opts_row.addWidget(self._chk_case)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._status)

        # Results list + preview
        splitter = QSplitter(Qt.Vertical)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection)
        splitter.addWidget(self._list)

        self._preview = _CropPreview()
        splitter.addWidget(self._preview)
        splitter.setSizes([120, 180])
        layout.addWidget(splitter, 1)

        # Padding
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("切图边距 (px):"))
        self._padding = QSpinBox()
        self._padding.setRange(0, 60)
        self._padding.setValue(4)
        pad_row.addWidget(self._padding)
        pad_row.addStretch()
        layout.addLayout(pad_row)

        # Save button
        self._btn_save = QPushButton("保存选中切图…")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_crop)
        layout.addWidget(self._btn_save)

    # ------------------------------------------------------------------

    def _do_search(self):
        doc = self._state.active_document
        if doc is None:
            self._status.setText("没有加载的文档")
            return
        query = self._query.text().strip()
        if not query:
            self._status.setText("请输入查找文字")
            return

        self._matches = find_text_matches(
            doc, query,
            case_sensitive=self._chk_case.isChecked(),
            search_chars=self._chk_chars.isChecked(),
            search_lines=self._chk_lines.isChecked(),
        )

        self._list.clear()
        self._preview.clear()
        self._btn_save.setEnabled(False)

        if not self._matches:
            self._status.setText("未找到匹配")
            return

        self._status.setText(f"找到 {len(self._matches)} 个匹配")
        for m in self._matches:
            bbox_str = ""
            if m.bbox:
                b = m.bbox
                bbox_str = f"  [{b.x:.0f},{b.y:.0f},{b.w:.0f}×{b.h:.0f}]"
            # Show bbox_source for chars
            src = ""
            if m.kind == "char":
                src = getattr(m.node, "bbox_source", "")
                gran = getattr(m.node, "bbox_granularity", "")
                src = f"  {src}/{gran}"
            label = f"[{m.kind}] {m.match_text[:40]}{bbox_str}{src}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, len(self._list) + len(self._matches) * 0)  # index
            # Show char/line differently
            if m.kind == "char":
                item.setForeground(Qt.darkGray)
            self._list.addItem(item)

        self._list.setCurrentRow(0)

    def _on_selection(self, current: QListWidgetItem, _prev):
        if current is None:
            self._preview.clear()
            self._btn_save.setEnabled(False)
            return

        idx = self._list.row(current)
        if idx < 0 or idx >= len(self._matches):
            return

        match = self._matches[idx]
        if not match.image_path or not match.bbox:
            self._preview.show_error("没有图像路径或 bbox")
            self._btn_save.setEnabled(False)
            return

        self._preview.show_loading()
        self._btn_save.setEnabled(False)

        padding = self._padding.value()
        worker = _CropWorker(match.image_path, match.bbox, padding=padding)
        worker.done.connect(lambda pix: self._on_crop_done(pix, match))
        worker.error.connect(self._preview.show_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_crop_done(self, pix: QPixmap, match: TextMatch):
        b = match.bbox
        bbox_str = f"x={b.x:.0f} y={b.y:.0f} w={b.w:.0f} h={b.h:.0f}" if b else ""
        src = getattr(match.node, "bbox_source", "")
        gran = getattr(match.node, "bbox_granularity", "")
        info = f"{match.kind} | {bbox_str} | {src}/{gran}"
        self._preview.show_pixmap(pix, info)
        self._btn_save.setEnabled(True)

    def _save_crop(self):
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._matches):
            return
        match = self._matches[idx]
        if not match.image_path or not match.bbox:
            return

        # Suggest filename from text
        safe = "".join(c for c in match.match_text[:20] if c.isalnum() or c in "-_")
        default = str(Path(match.image_path).stem) + f"__{safe}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存切图", default,
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;All Files (*)",
        )
        if not path:
            return
        try:
            save_crop(match.image_path, match.bbox, path, padding=self._padding.value())
            self._status.setText(f"已保存: {Path(path).name}")
        except Exception as exc:
            self._status.setText(f"保存失败: {exc}")


# ---------------------------------------------------------------------------
# Manual crop tab
# ---------------------------------------------------------------------------

class _ManualCropTab(QWidget):
    """手动切图: 输入坐标 → 切图 → 保存."""

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        self._worker: Optional[_CropWorker] = None

        layout = QVBoxLayout(self)

        grp = QGroupBox("坐标（图像像素空间）")
        form = QHBoxLayout(grp)
        self._x = self._make_spinbox(form, "x:")
        self._y = self._make_spinbox(form, "y:")
        self._w = self._make_spinbox(form, "w:")
        self._h = self._make_spinbox(form, "h:")
        layout.addWidget(grp)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("边距:"))
        self._padding = QSpinBox()
        self._padding.setRange(0, 60)
        self._padding.setValue(0)
        pad_row.addWidget(self._padding)
        pad_row.addStretch()
        layout.addLayout(pad_row)

        btn_row = QHBoxLayout()
        self._btn_crop = QPushButton("切图预览")
        self._btn_crop.clicked.connect(self._do_crop)
        btn_row.addWidget(self._btn_crop)
        self._btn_save = QPushButton("保存…")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save)
        btn_row.addWidget(self._btn_save)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._status)

        self._preview = _CropPreview()
        layout.addWidget(self._preview, 1)

        self._last_pil = None

    @staticmethod
    def _make_spinbox(row, label):
        row.addWidget(QLabel(label))
        sb = QSpinBox()
        sb.setRange(0, 99999)
        sb.setValue(0)
        row.addWidget(sb)
        return sb

    def set_from_bbox(self, bbox: BBox):
        """Called when canvas selection changes: pre-fill coords."""
        self._x.setValue(int(bbox.x))
        self._y.setValue(int(bbox.y))
        self._w.setValue(int(bbox.w))
        self._h.setValue(int(bbox.h))

    def _do_crop(self):
        page = self._state.active_page
        if page is None or not page.image_path:
            self._status.setText("没有图像")
            return
        bbox = BBox(
            x=self._x.value(), y=self._y.value(),
            w=self._w.value(), h=self._h.value(),
        )
        if bbox.w <= 0 or bbox.h <= 0:
            self._status.setText("w / h 必须大于 0")
            return

        self._preview.show_loading()
        self._btn_save.setEnabled(False)

        worker = _CropWorker(page.image_path, bbox, padding=self._padding.value())
        worker.done.connect(self._on_done)
        worker.error.connect(self._preview.show_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, pix: QPixmap):
        self._preview.show_pixmap(pix)
        self._btn_save.setEnabled(True)

    def _save(self):
        page = self._state.active_page
        if page is None or not page.image_path:
            return
        bbox = BBox(
            x=self._x.value(), y=self._y.value(),
            w=self._w.value(), h=self._h.value(),
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "保存切图", f"crop_{bbox.x:.0f}_{bbox.y:.0f}.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;All Files (*)",
        )
        if not path:
            return
        try:
            save_crop(page.image_path, bbox, path, padding=self._padding.value())
            self._status.setText(f"已保存: {Path(path).name}")
        except Exception as exc:
            self._status.setText(f"保存失败: {exc}")


# ---------------------------------------------------------------------------
# Public CropPanel (exported to main window)
# ---------------------------------------------------------------------------

class CropPanel(QWidget):
    """文本找图 & 切图面板（主窗口右侧 tab）."""

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._search_tab = _TextSearchTab(state)
        self._manual_tab = _ManualCropTab(state)
        self._tabs.addTab(self._search_tab, "文本找图")
        self._tabs.addTab(self._manual_tab, "手动切图")
        layout.addWidget(self._tabs)

        # When a bbox is selected on canvas, pre-fill manual tab
        state.on_node_selected.connect(self._on_node_selected)

    def _on_node_selected(self, node):
        if node is None:
            return
        bbox = getattr(node, "bbox", None)
        if bbox is not None:
            self._manual_tab.set_from_bbox(bbox)

    def focus_search(self, query: str = ""):
        """Switch to text-search tab and optionally pre-fill query."""
        self._tabs.setCurrentIndex(0)
        if query:
            self._search_tab._query.setText(query)
            self._search_tab._do_search()
