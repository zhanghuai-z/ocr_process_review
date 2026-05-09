"""Run OCR panel — lets the user call PaddleOCR directly on the current image.

Supports:
  * PaddleOCR (text OCR, PaddleOCR 3.x)
  * PPStructureV3 (layout + OCR)

Parameters are exposed as editable controls.  Results are parsed through the
existing PaddleAdapter and loaded into AppState.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from tools.ocr_inspector.adapters.paddle import PaddleAdapter
from tools.ocr_inspector.state import AppState


# ---------------------------------------------------------------------------
# Worker signal carrier (must be a QObject to use signals from a bg thread)
# ---------------------------------------------------------------------------

class _Signals(QObject):
    finished = Signal(object, str)   # (raw_result_dict, image_path)
    error    = Signal(str)


# ---------------------------------------------------------------------------
# Result flatteners
# ---------------------------------------------------------------------------

def _flatten_paddle_result(result_list: list) -> dict:
    """Convert PaddleOCR 3.x predict() list to the dict shape PaddleAdapter expects."""
    merged: dict[str, Any] = {}
    for item in result_list:
        if hasattr(item, "json"):
            d = item.json()
            if "overall_ocr_res" in d:
                merged.setdefault("overall_ocr_res", {})
                for k, v in d["overall_ocr_res"].items():
                    if isinstance(v, list):
                        merged["overall_ocr_res"].setdefault(k, [])
                        merged["overall_ocr_res"][k].extend(v)
                    else:
                        merged["overall_ocr_res"][k] = v
            if "parsing_res_list" in d:
                merged.setdefault("parsing_res_list", [])
                merged["parsing_res_list"].extend(d["parsing_res_list"])
    if not merged and result_list and hasattr(result_list[0], "json"):
        return result_list[0].json()
    return merged


def _flatten_structure_result(result_list: list) -> dict:
    """Convert PPStructureV3 predict() output to raw dict for PaddleAdapter."""
    out: dict[str, Any] = {"parsing_res_list": []}
    for item in result_list:
        if hasattr(item, "json"):
            d = item.json()
            if "parsing_res_list" in d:
                out["parsing_res_list"].extend(d["parsing_res_list"])
            elif isinstance(d, dict):
                out["parsing_res_list"].append(d)
    return out


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class RunOcrPanel(QWidget):
    """Controls panel for running PaddleOCR live on an image."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._signals = _Signals()
        self._signals.finished.connect(self._on_ocr_finished)
        self._signals.error.connect(self._on_ocr_error)
        self._last_raw: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── image path ──────────────────────────────────────────────────────
        img_group = QGroupBox("Image")
        img_layout = QHBoxLayout(img_group)
        self._img_edit = QLineEdit()
        self._img_edit.setPlaceholderText("(uses active page image)")
        self._img_btn = QPushButton("…")
        self._img_btn.setFixedWidth(28)
        self._img_btn.clicked.connect(self._browse_image)
        img_layout.addWidget(self._img_edit)
        img_layout.addWidget(self._img_btn)
        layout.addWidget(img_group)

        # ── model selection ──────────────────────────────────────────────────
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItem("PaddleOCR (文字检测+识别)", "paddleocr")
        self._model_combo.addItem("PPStructureV3 (版面分析)", "ppstructure")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_form.addRow("Pipeline:", self._model_combo)

        self._lang_combo = QComboBox()
        for lang in ["ch", "en", "japan", "korean", "fr", "de", "ar"]:
            self._lang_combo.addItem(lang)
        model_form.addRow("Language:", self._lang_combo)

        self._ocr_version_combo = QComboBox()
        self._ocr_version_combo.addItem("PP-OCRv5 (default)", "")
        self._ocr_version_combo.addItem("PP-OCRv4", "PP-OCRv4")
        self._ocr_version_combo.addItem("PP-OCRv3", "PP-OCRv3")
        model_form.addRow("OCR version:", self._ocr_version_combo)
        layout.addWidget(model_group)

        # ── preprocessing ────────────────────────────────────────────────────
        pre_group = QGroupBox("预处理")
        pre_form = QFormLayout(pre_group)
        self._use_orientation = QCheckBox()
        self._use_orientation.setChecked(False)
        pre_form.addRow("文档方向校正:", self._use_orientation)

        self._use_unwarping = QCheckBox()
        self._use_unwarping.setChecked(False)
        pre_form.addRow("图像去扭曲:", self._use_unwarping)

        self._use_textline_orient = QCheckBox()
        self._use_textline_orient.setChecked(False)
        pre_form.addRow("文本行方向:", self._use_textline_orient)
        layout.addWidget(pre_group)

        # ── det params ───────────────────────────────────────────────────────
        det_group = QGroupBox("检测参数 (det)")
        det_form = QFormLayout(det_group)

        self._det_thresh = QDoubleSpinBox()
        self._det_thresh.setRange(0.01, 1.0)
        self._det_thresh.setSingleStep(0.05)
        self._det_thresh.setDecimals(2)
        self._det_thresh.setValue(0.3)
        det_form.addRow("det_thresh:", self._det_thresh)

        self._det_box_thresh = QDoubleSpinBox()
        self._det_box_thresh.setRange(0.01, 1.0)
        self._det_box_thresh.setSingleStep(0.05)
        self._det_box_thresh.setDecimals(2)
        self._det_box_thresh.setValue(0.6)
        det_form.addRow("det_box_thresh:", self._det_box_thresh)

        self._det_unclip_ratio = QDoubleSpinBox()
        self._det_unclip_ratio.setRange(0.5, 5.0)
        self._det_unclip_ratio.setSingleStep(0.1)
        self._det_unclip_ratio.setDecimals(2)
        self._det_unclip_ratio.setValue(1.5)
        det_form.addRow("det_unclip_ratio:", self._det_unclip_ratio)

        self._det_limit_side_len = QSpinBox()
        self._det_limit_side_len.setRange(64, 4096)
        self._det_limit_side_len.setSingleStep(64)
        self._det_limit_side_len.setValue(736)
        det_form.addRow("det_limit_side_len:", self._det_limit_side_len)

        self._det_limit_type_combo = QComboBox()
        self._det_limit_type_combo.addItem("max")
        self._det_limit_type_combo.addItem("min")
        det_form.addRow("det_limit_type:", self._det_limit_type_combo)
        layout.addWidget(det_group)

        # ── rec params ───────────────────────────────────────────────────────
        rec_group = QGroupBox("识别参数 (rec)")
        rec_form = QFormLayout(rec_group)

        self._rec_score_thresh = QDoubleSpinBox()
        self._rec_score_thresh.setRange(0.0, 1.0)
        self._rec_score_thresh.setSingleStep(0.05)
        self._rec_score_thresh.setDecimals(2)
        self._rec_score_thresh.setValue(0.0)
        rec_form.addRow("rec_score_thresh:", self._rec_score_thresh)

        self._return_word_box = QCheckBox()
        self._return_word_box.setChecked(False)
        rec_form.addRow("return_word_box:", self._return_word_box)
        layout.addWidget(rec_group)

        # ── layout params (PPStructureV3 only) ────────────────────────────────
        self._layout_group = QGroupBox("版面参数 (layout)")
        layout_form = QFormLayout(self._layout_group)

        self._layout_thresh = QDoubleSpinBox()
        self._layout_thresh.setRange(0.01, 1.0)
        self._layout_thresh.setSingleStep(0.05)
        self._layout_thresh.setDecimals(2)
        self._layout_thresh.setValue(0.5)
        layout_form.addRow("layout_threshold:", self._layout_thresh)

        self._layout_nms = QCheckBox()
        self._layout_nms.setChecked(True)
        layout_form.addRow("layout_nms:", self._layout_nms)

        self._layout_unclip = QDoubleSpinBox()
        self._layout_unclip.setRange(0.5, 5.0)
        self._layout_unclip.setSingleStep(0.1)
        self._layout_unclip.setDecimals(2)
        self._layout_unclip.setValue(1.0)
        layout_form.addRow("layout_unclip_ratio:", self._layout_unclip)
        layout.addWidget(self._layout_group)
        self._layout_group.setVisible(False)

        # ── run / save buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run OCR")
        self._run_btn.setStyleSheet("font-weight:bold; padding:4px 8px;")
        self._run_btn.clicked.connect(self._run_ocr)
        btn_row.addWidget(self._run_btn)

        self._save_btn = QPushButton("Save JSON…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_result)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

        # ── log ──────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        self._log.setStyleSheet("font-size:11px; font-family:monospace;")
        log_layout.addWidget(self._log)
        layout.addWidget(log_group)

        layout.addStretch()

    # ── slots ──────────────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All files (*)"
        )
        if path:
            self._img_edit.setText(path)

    def _on_model_changed(self, _index: int) -> None:
        is_structure = self._model_combo.currentData() == "ppstructure"
        self._layout_group.setVisible(is_structure)
        self._lang_combo.setEnabled(not is_structure)

    def _run_ocr(self) -> None:
        image_path = self._img_edit.text().strip()
        if not image_path:
            page = self._state.active_page
            if page and page.image_path:
                image_path = page.image_path
        if not image_path or not Path(image_path).is_file():
            QMessageBox.warning(self, "No image",
                                "Please specify a valid image file first.\n"
                                "(Set an image via the toolbar 'Set Image…' button, "
                                "or type a path above.)")
            return

        self._run_btn.setEnabled(False)
        self._log.clear()
        self._log.append(f"Running OCR on: {image_path}")

        params = self._collect_params()
        model_key = self._model_combo.currentData()
        signals = self._signals

        def worker() -> None:
            try:
                if model_key == "ppstructure":
                    from paddleocr import PPStructureV3
                    engine = PPStructureV3(**params["structure"])
                    result = list(engine.predict(image_path))
                    raw = _flatten_structure_result(result)
                else:
                    from paddleocr import PaddleOCR
                    init_p = {k: v for k, v in params["ocr_init"].items() if v is not None}
                    pred_p = {k: v for k, v in params["ocr_pred"].items() if v is not None}
                    engine = PaddleOCR(**init_p)
                    result = list(engine.predict(image_path, **pred_p))
                    raw = _flatten_paddle_result(result)
                signals.finished.emit(raw, image_path)
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _collect_params(self) -> dict:
        lang = self._lang_combo.currentText()
        ver = self._ocr_version_combo.currentData() or None

        ocr_init: dict[str, Any] = {
            "lang": lang,
            "ocr_version": ver,
            "use_doc_orientation_classify": self._use_orientation.isChecked(),
            "use_doc_unwarping": self._use_unwarping.isChecked(),
            "use_textline_orientation": self._use_textline_orient.isChecked(),
        }
        ocr_pred: dict[str, Any] = {
            "text_det_thresh": self._det_thresh.value(),
            "text_det_box_thresh": self._det_box_thresh.value(),
            "text_det_unclip_ratio": self._det_unclip_ratio.value(),
            "text_det_limit_side_len": self._det_limit_side_len.value(),
            "text_det_limit_type": self._det_limit_type_combo.currentText(),
            "text_rec_score_thresh": self._rec_score_thresh.value(),
            "return_word_box": self._return_word_box.isChecked(),
        }
        structure: dict[str, Any] = {
            "text_det_thresh": self._det_thresh.value(),
            "text_det_box_thresh": self._det_box_thresh.value(),
            "text_det_unclip_ratio": self._det_unclip_ratio.value(),
            "text_det_limit_side_len": self._det_limit_side_len.value(),
            "text_det_limit_type": self._det_limit_type_combo.currentText(),
            "text_rec_score_thresh": self._rec_score_thresh.value(),
            "layout_threshold": self._layout_thresh.value(),
            "layout_nms": self._layout_nms.isChecked(),
            "layout_unclip_ratio": self._layout_unclip.value(),
        }
        return {"ocr_init": ocr_init, "ocr_pred": ocr_pred, "structure": structure}

    def _on_ocr_finished(self, raw: dict, image_path: str) -> None:
        self._run_btn.setEnabled(True)
        self._last_raw = raw
        self._save_btn.setEnabled(True)
        self._log.append("OCR finished — parsing result…")
        try:
            doc = PaddleAdapter().parse(raw, source_path="<live>", image_path=image_path)
        except Exception as exc:
            self._log.append(f"[ERROR] Adapter parse failed: {exc}")
            return
        n_lines = sum(len(list(p.all_lines)) for p in doc.pages)
        n_blocks = sum(len(p.blocks) for p in doc.pages)
        self._log.append(
            f"OK — {len(doc.pages)} page(s), {n_blocks} block(s), {n_lines} line(s)"
        )
        for msg in (doc.parse_log or [])[:5]:
            self._log.append(f"  warn: {msg}")
        self._state.set_document(doc)

    def _on_ocr_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._log.append(f"[ERROR] {msg}")

    def _save_result(self) -> None:
        if self._last_raw is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save OCR result", "ocr_result.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._last_raw, fh, ensure_ascii=False, indent=2)
            self._log.append(f"Saved → {path}")
        except Exception as exc:
            self._log.append(f"[ERROR] save failed: {exc}")
