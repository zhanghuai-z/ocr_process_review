"""OCR Inspector main window."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QScrollArea, QSplitter,
    QStatusBar, QTabWidget, QToolBar, QComboBox,
)

from tools.ocr_inspector.adapters import registry as adapter_registry
from tools.ocr_inspector.models.ir import DocumentNode, PageNode
from tools.ocr_inspector.state import AppState
from tools.ocr_inspector.ui.canvas import OcrCanvas
from tools.ocr_inspector.ui.panels import JsonTreePanel
from tools.ocr_inspector.ui.panels.inspector import InspectorPanel
from tools.ocr_inspector.ui.panels.run_ocr import RunOcrPanel
from tools.ocr_inspector.ui.panels.params_ref import ParamsRefPanel
from tools.ocr_inspector.ui.panels.parse_log import ParseLogPanel
from tools.ocr_inspector.ui.panels.crop_panel import CropPanel


class OcrInspectorWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCR Inspector")
        self.resize(1400, 900)

        self._state = AppState()
        self._tree   = JsonTreePanel(self._state)
        self._canvas = OcrCanvas(self._state)
        self._insp   = InspectorPanel(self._state)
        self._run    = RunOcrPanel(self._state)

        # Right-side tabs: Inspector + Run OCR
        right_tabs = QTabWidget()
        right_tabs.addTab(self._insp, "Inspector")
        run_scroll = QScrollArea()
        run_scroll.setWidget(self._run)
        run_scroll.setWidgetResizable(True)
        run_scroll.setMinimumWidth(280)
        right_tabs.addTab(run_scroll, "Run OCR")
        self._parse_log = ParseLogPanel(self._state)
        right_tabs.addTab(self._parse_log, "解析日志")
        self._params_ref = ParamsRefPanel()
        right_tabs.addTab(self._params_ref, "参数说明")
        self._crop = CropPanel(self._state)
        right_tabs.addTab(self._crop, "文本找图")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._canvas)
        splitter.addWidget(right_tabs)
        splitter.setSizes([280, 820, 320])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_act = QAction("Open JSON\u2026", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_file)
        toolbar.addAction(open_act)

        open_img_act = QAction("Set Image\u2026", self)
        open_img_act.triggered.connect(self._set_image)
        toolbar.addAction(open_img_act)

        open_only_img_act = QAction("Open Image\u2026", self)
        open_only_img_act.setShortcut(QKeySequence("Ctrl+I"))
        open_only_img_act.setToolTip("\u76f4\u63a5\u6253\u5f00\u56fe\u7247\uff08\u4e0d\u9700\u8981 JSON\uff09\uff0c\u53ef\u5728 Run OCR \u9762\u677f\u8fd0\u884c\u8bc6\u522b")
        open_only_img_act.triggered.connect(self._open_image_only)
        toolbar.addAction(open_only_img_act)

        ocr_cfg_act = QAction("OCR Settings…", self)
        ocr_cfg_act.triggered.connect(self._show_ocr_settings)
        toolbar.addAction(ocr_cfg_act)

        toolbar.addSeparator()

        self._page_combo = QComboBox()
        self._page_combo.setMinimumWidth(120)
        self._page_combo.currentIndexChanged.connect(self._on_page_index_changed)
        toolbar.addWidget(QLabel("  Page: "))
        toolbar.addWidget(self._page_combo)

        toolbar.addSeparator()

        for key, label, shortcut in [
            ("blocks",   "Blocks",   "F1"),
            ("lines",    "Lines",    "F2"),
            ("chars",    "Chars",    "F3"),
            ("polygons", "Polygons", "F4"),
            ("labels",   "Labels",   "F5"),
            ("layout_det", "LayoutDet", "F6"),
            ("legend",   "Legend",   "F7"),
        ]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(self._state.overlay_flags.get(key, True))
            act.setShortcut(QKeySequence(shortcut))
            act.toggled.connect(lambda checked, k=key: self._state.set_overlay(k, checked))
            toolbar.addAction(act)

        self._status_image  = QLabel("image: \u2014")
        self._status_scene  = QLabel("scene: \u2014")
        self._status_engine = QLabel("engine: \u2014")
        status = QStatusBar()
        status.addPermanentWidget(self._status_image)
        status.addPermanentWidget(self._status_scene)
        status.addPermanentWidget(self._status_engine)
        self.setStatusBar(status)

        self._canvas.coord_changed.connect(self._on_coord_changed)
        self._state.on_document_changed(self._on_doc_loaded)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OCR JSON", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self.load_file(path)

    def _on_doc_loaded(self, doc) -> None:
        """Called whenever AppState.set_document() is called (file load OR Run OCR)."""
        if doc is None:
            self._page_combo.clear()
            return
        self._page_combo.blockSignals(True)
        self._page_combo.clear()
        for page in doc.pages:
            self._page_combo.addItem(f"Page {page.page_number}", page)
        self._page_combo.setCurrentIndex(0)
        self._page_combo.blockSignals(False)
        self._status_engine.setText(f"engine: {doc.engine}")
        if doc.pages:
            self._state.set_active_page(doc.pages[0])

    def load_file(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading {path}: {exc}", 5000)
            return

        adapter_cls = adapter_registry.auto_detect(raw) or adapter_registry.get("paddle")
        image_path = self._guess_image_path(path)
        doc = adapter_cls().parse(raw, source_path=path, image_path=image_path or "")

        n_warn = sum(1 for m in (doc.parse_log or []) if "WARNING" in str(m) or "ERROR" in str(m))
        if n_warn:
            self.statusBar().showMessage(f"Loaded · {n_warn} warnings/errors — see 解析日志 tab", 5000)
        else:
            self.statusBar().showMessage(f"Loaded: {Path(path).name}", 3000)
        self._state.set_document(doc)   # _on_doc_loaded fires here

    def _open_image_only(self) -> None:
        """Open image without JSON — creates stub doc visible immediately."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
        )
        if not path:
            return
        doc = DocumentNode.make(source_path=path, engine="image-only")
        page = PageNode.make(page_number=1, image_path=path)
        doc.pages.append(page)
        doc.parse_log.append(f"INFO: image-only mode \u2014 {Path(path).name}")
        self._state.set_document(doc)
        self.statusBar().showMessage(f"Opened image: {Path(path).name}", 3000)

    def _set_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Set image for current page", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
        )
        if not path:
            return
        page = self._state.active_page
        if page is not None:
            page.image_path = path
            self._state.set_active_page(page)
        else:
            # No document loaded yet — create stub so image is visible
            doc = DocumentNode.make(source_path=path, engine="image-only")
            stub = PageNode.make(page_number=1, image_path=path)
            doc.pages.append(stub)
            doc.parse_log.append(f"INFO: image-only mode \u2014 {Path(path).name}")
            self._state.set_document(doc)
            self.statusBar().showMessage(f"Image set: {Path(path).name}", 3000)

    @staticmethod
    def _guess_image_path(json_path: str) -> Optional[str]:
        stem = Path(json_path).stem
        parent = Path(json_path).parent
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            candidate = parent / (stem + ext)
            if candidate.exists():
                return str(candidate)
        return None

    def _show_ocr_settings(self) -> None:
        try:
            from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        except ImportError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "OCR Settings", f"API settings dialog not available:\n{exc}")
            return
        dlg = ApiSettingsDialog(self)
        if dlg.exec():
            self._run.refresh_api_summary()

    def _on_page_index_changed(self, index: int) -> None:
        page = self._page_combo.itemData(index)
        if isinstance(page, PageNode):
            self._state.set_active_page(page)

    def _on_coord_changed(self, ix: float, iy: float, sx: float, sy: float) -> None:
        self._status_image.setText(f"image: ({ix:.0f}, {iy:.0f})")
        self._status_scene.setText(f"scene: ({sx:.0f}, {sy:.0f})")
